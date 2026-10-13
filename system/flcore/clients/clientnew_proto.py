from collections import defaultdict
import copy
import os
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

from flcore.clients.clientnew import clientNew


class clientNewProto(clientNew):
    """
    Client cho FedNewProto:
    Kế thừa clientNew (FedNew), tích hợp Prototypical Contrastive Loss theo cách tính của FPL
    (kết hợp InfoNCE và MSE tới global prototype của từng class).
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        if not hasattr(self, 'args') or self.args is None:
            self.args = args

        # 1.1 Tham số nhiệt độ InfoNCE và hệ số điều chuẩn Prototypical Contrastive Loss
        v_temp = getattr(args, 'infoNCET', 0.1)
        self.infoNCET = 0.1 if v_temp is None else float(v_temp)

        v_proto = getattr(args, 'lamda_proto', 1.0)
        self.lamda_proto = 1.0 if v_proto is None else float(v_proto)

        # 1.2 Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # 1.3 Cấu hình Optimizer an toàn
        encoder_lr = getattr(args, 'encoder_lr', None)
        head_params = list(self.head.parameters())

        if encoder_lr is None or encoder_lr <= 0 or len(head_params) == 0:
            # Trường hợp mặc định: giữ nguyên self.optimizer gốc cho toàn model, không tạo optimizer thứ hai
            self.opt_encoder = None
            self.opt_encoder_scheduler = None

            # Đảm bảo self.optimizer bao phủ toàn bộ tham số model (phòng ngừa trường hợp lớp cha đã tách)
            all_model_params = set(self.model.parameters())
            current_opt_params = set(p for group in self.optimizer.param_groups for p in group['params'])
            if current_opt_params != all_model_params:
                opt_cls = type(self.optimizer)
                cfg = copy.deepcopy(self.optimizer.defaults)
                cfg['lr'] = self.learning_rate
                self.optimizer = opt_cls(self.model.parameters(), **cfg)
                if self.learning_rate_decay:
                    sched_cls = type(self.learning_rate_scheduler) if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None else torch.optim.lr_scheduler.ExponentialLR
                    gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                    self.learning_rate_scheduler = sched_cls(self.optimizer, gamma=gamma)
        else:
            # Chỉ khi encoder_lr hợp lệ (> 0) và head có tham số:
            opt_cls = type(self.optimizer)
            def_cfg = copy.deepcopy(self.optimizer.defaults)
            enc_cfg, head_cfg = copy.deepcopy(def_cfg), copy.deepcopy(def_cfg)
            enc_cfg['lr'] = float(encoder_lr)
            head_cfg['lr'] = self.learning_rate

            self.opt_encoder = opt_cls(self.encoder.parameters(), **enc_cfg)
            self.optimizer = opt_cls(self.head.parameters(), **head_cfg)

            opt_enc_params = set(p for group in self.opt_encoder.param_groups for p in group['params'])
            opt_head_params = set(p for group in self.optimizer.param_groups for p in group['params'])
            assert opt_enc_params.isdisjoint(opt_head_params), "Tham số của 2 optimizer không được trùng lặp!"
            assert (opt_enc_params | opt_head_params) == set(self.model.parameters()), "Hợp tham số 2 optimizer phải bằng toàn bộ tham số model!"

            if self.learning_rate_decay:
                sched_cls = type(self.learning_rate_scheduler) if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None else torch.optim.lr_scheduler.ExponentialLR
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.learning_rate_scheduler = sched_cls(self.optimizer, gamma=gamma)
                self.opt_encoder_scheduler = sched_cls(self.opt_encoder, gamma=gamma)
            else:
                self.opt_encoder_scheduler = None

    def _split_model(self):
        """
        Tách backbone (encoder trích xuất đặc trưng) và head phân lớp:
        - Ưu tiên 1: model.base / model.head (BaseHeadSplit)
        - Ưu tiên 2: model.features / model.classifier
        - Fallback: encoder = model, head = nn.Identity() (khi đó rep chính là logits)
        """
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            encoder, head = self.model.base, self.model.head
        elif hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            encoder, head = self.model.features, self.model.classifier
        else:
            # Fallback: toàn bộ model đóng vai trò encoder, head là Identity
            # Lưu ý: Trong trường hợp fallback này, representation (rep) chính là logits.
            encoder, head = self.model, nn.Identity()

        enc_params = set(encoder.parameters())
        head_params = set(head.parameters())
        model_params = set(self.model.parameters())

        assert enc_params.isdisjoint(head_params), "Tham số của encoder và head phải rời nhau!"
        assert (enc_params | head_params) == model_params, "Hợp tham số encoder và head phải bằng toàn bộ tham số model!"

        return encoder, head

    def extract_features(self, x):
        """
        Trích xuất biểu diễn đặc trưng rep và logits out:
        - Nếu x là list: lấy x[0] đưa lên self.device; ngược lại x.to(self.device).
        - rep_raw = self.encoder(x); out = self.head(rep_raw).
        - rep = rep_raw.view(rep_raw.size(0), -1) để dùng cho loss InfoNCE + MSE.
        - Trả về tuple (rep, out).
        """
        x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        rep_raw = self.encoder(x)
        try:
            out = self.head(rep_raw)
        except (RuntimeError, ValueError):
            out = self.head(rep_raw.view(rep_raw.size(0), -1))
        rep = rep_raw.view(rep_raw.size(0), -1)
        return rep, out

    def calculate_infonce(self, f_now, f_pos, f_neg, pos_mask=None):
        """
        Tính InfoNCE Loss theo chuẩn FPL:
        - f_proto = torch.cat((f_pos, f_neg), dim=0)
        - l = torch.cosine_similarity(f_now, f_proto, dim=1) / self.infoNCET
        - exp_l = torch.exp(l).view(1, -1)
        - pos_mask: nếu truyền từ cache thì dùng luôn; nếu None thì tạo dự phòng
        - infonce_loss = -torch.log(pos_l.sum(1) / exp_l.sum(1)).squeeze()
        """
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1) / self.infoNCET
        exp_l = torch.exp(l).view(1, -1)

        if pos_mask is None:
            pos_mask = torch.tensor(
                [1.0] * f_pos.size(0) + [0.0] * f_neg.size(0),
                dtype=torch.float,
                device=self.device
            ).view(1, -1)

        pos_l = exp_l * pos_mask
        infonce_loss = -torch.log(pos_l.sum(1) / exp_l.sum(1)).squeeze()
        return infonce_loss

    def hierarchical_info_loss(self, f_now, label_val, cache):
        """
        Tính Hierarchical Info Loss kết hợp InfoNCE và MSE tới global prototype:
        - label_val: giá trị int của nhãn lớp
        - cache: dict tiền xử lý chứa all_f, f_neg, mean_f, pos_mask
        - Trả về scalar 0-D chuẩn PyTorch: (xi_info_loss + cu_info_loss).view(())
        """
        f_pos = cache['all_f'][label_val]
        f_neg = cache['f_neg'][label_val]
        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg, cache['pos_mask'][label_val])
        mean_f_pos = cache['mean_f'][label_val]
        cu_info_loss = nn.MSELoss()(f_now, mean_f_pos)
        return (xi_info_loss + cu_info_loss).view(())

    def _build_proto_cache(self):
        """
        Tiền xử lý prototype cache trên self.device trước vòng epoch:
        - Lọc class hợp lệ: norm > 1e-8.
        - Nếu số class hợp lệ < 2: trả về None (không đủ f_neg để tính contrastive).
        - Đóng gói all_f, mean_f, f_neg, pos_mask sẵn trên self.device dạng 2D (1, D).
        """
        if not hasattr(self, 'global_prototypes') or not self.global_prototypes:
            return None

        valid_classes = sorted([
            int(k) for k in self.global_prototypes.keys()
            if self.global_prototypes[k] is not None and torch.norm(self.global_prototypes[k].float()) > 1e-8
        ])

        if len(valid_classes) < 2:
            return None

        # Mỗi class có đúng 1 prototype = global mean, shape (1, D)
        all_f = {
            c: self.global_prototypes[c].detach().float().view(1, -1).to(self.device)
            for c in valid_classes
        }
        mean_f = {
            c: self.global_prototypes[c].detach().float().view(1, -1).to(self.device)
            for c in valid_classes
        }
        f_neg = {
            c: torch.cat([all_f[other] for other in valid_classes if other != c], dim=0)
            for c in valid_classes
        }
        pos_mask = {
            c: torch.tensor(
                [1.0] * all_f[c].size(0) + [0.0] * f_neg[c].size(0),
                dtype=torch.float,
                device=self.device
            ).view(1, -1)
            for c in valid_classes
        }

        return {
            'valid_classes': set(valid_classes),
            'all_f': all_f,
            'mean_f': mean_f,
            'f_neg': f_neg,
            'pos_mask': pos_mask
        }

    def train(self):
        """
        Huấn luyện cục bộ:
        - CE Loss trên logits phân lớp
        - Prototypical Contrastive Loss (InfoNCE + MSE) trên feature representations
        - Cập nhật thống kê cục bộ (mean, std, count) sau epoch
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        cache = self._build_proto_cache()
        max_local_steps = np.random.randint(1, max(2, self.local_epochs // 2 + 1)) if self.train_slow else self.local_epochs

        ce_losses = []
        proto_losses = []

        for _ in range(max_local_steps):
            for x, y in trainloader:
                if isinstance(x, list):
                    x = [item.to(self.device) if isinstance(item, torch.Tensor) else item for item in x]
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep, out = self.extract_features(x)
                loss_ce = self.loss(out, y)

                if cache is None:
                    loss_InfoNCE = 0.0 * loss_ce
                else:
                    # Kiểm tra tính khớp chiều feature D với prototype
                    proto_dim = next(iter(cache['all_f'].values())).size(1)
                    assert rep.size(1) == proto_dim, f"Feature dimension mismatch: rep has {rep.size(1)}, prototype has {proto_dim}"

                    loss_InfoNCE = None
                    for i in range(len(y)):
                        label_val = int(y[i].item())
                        if label_val in cache['valid_classes']:
                            f_now = rep[i].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(f_now, label_val, cache)
                            loss_InfoNCE = loss_instance if (loss_InfoNCE is None) else (loss_InfoNCE + loss_instance)

                    # Chuẩn FPL: chia cho tổng số mẫu batch len(y), không chia cho số mẫu hợp lệ
                    if loss_InfoNCE is None:
                        loss_InfoNCE = 0.0 * loss_ce
                    else:
                        loss_InfoNCE = loss_InfoNCE / len(y)

                loss_total = loss_ce + self.lamda_proto * loss_InfoNCE

                self.optimizer.zero_grad()
                if self.opt_encoder is not None:
                    self.opt_encoder.zero_grad()

                loss_total.backward()

                self.optimizer.step()
                if self.opt_encoder is not None:
                    self.opt_encoder.step()

                ce_losses.append(loss_ce.item())
                proto_losses.append(loss_InfoNCE.item())

        # Ghi log WandB an toàn nếu cờ log được bật
        if getattr(self.args, 'log', False):
            try:
                if wandb.run is not None:
                    mean_ce = float(np.mean(ce_losses)) if ce_losses else 0.0
                    mean_proto = float(np.mean(proto_losses)) if proto_losses else 0.0
                    wandb.log({
                        f"clients/client_{self.id}_train_loss": mean_ce,
                        f"clients/client_{self.id}_loss_ce": mean_ce,
                        f"clients/client_{self.id}_loss_infonce": mean_proto
                    }, commit=False)
            except Exception:
                pass

        # Cập nhật mean, std, count chính xác gửi lên server
        self.extract_local_stats(trainloader)

        # Cập nhật learning rate scheduler nếu có
        if self.learning_rate_decay:
            if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None:
                self.learning_rate_scheduler.step()
            if hasattr(self, 'opt_encoder_scheduler') and self.opt_encoder_scheduler is not None:
                self.opt_encoder_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def visualize_tsne(self, round_idx=0, max_samples=1000, save_dir="./tsne_results"):
        """
        Trực quan hóa không gian đặc trưng bằng t-SNE:
        - Biểu diễn phân bố các mẫu test (tô màu theo label)
        - Điểm sao màu đen viền đỏ biểu thị global prototype của từng class
        - Tính Silhouette score định lượng độ tách biệt cụm trên đặc trưng gốc
        """
        self.model.eval()
        try:
            with torch.no_grad():
                testloader = self.load_test_data()
                all_reps = []
                all_labels = []

                for x, y in testloader:
                    x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                    rep_raw = self.encoder(x)
                    rep = rep_raw.view(rep_raw.size(0), -1)
                    all_reps.append(rep.cpu())
                    all_labels.append(y.cpu())

                if not all_reps:
                    print(f"[Client {self.id}] Cảnh báo: Tập kiểm thử rỗng, bỏ qua visualize_tsne.")
                    return None

                sample_feats = torch.cat(all_reps, dim=0).numpy()
                sample_labels = torch.cat(all_labels, dim=0).numpy()

                N = sample_feats.shape[0]
                if N > max_samples:
                    idx = np.random.RandomState(0).choice(N, max_samples, replace=False)
                    sample_feats = sample_feats[idx]
                    sample_labels = sample_labels[idx]

                n_samples = sample_feats.shape[0]
                unique_classes = np.unique(sample_labels)
                n_classes = len(unique_classes)

                # Silhouette score tính trên sample gốc (chưa qua t-SNE)
                sil = float(silhouette_score(sample_feats, sample_labels)) if (2 <= n_classes <= n_samples - 1) else None
                print(f"[Client {self.id}] Round {round_idx} - Silhouette: {sil}")

                # Gom global prototypes theo class đã sắp xếp
                proto_feats = []
                proto_classes = []
                if hasattr(self, 'global_prototypes') and self.global_prototypes:
                    valid_proto_classes = sorted([
                        int(k) for k, v in self.global_prototypes.items()
                        if v is not None and torch.norm(v.float()) > 1e-8
                    ])
                    for c in valid_proto_classes:
                        p_tensor = self.global_prototypes[c].detach().cpu().float().view(1, -1).numpy()
                        proto_feats.append(p_tensor)
                        proto_classes.append(c)

                has_protos = len(proto_feats) > 0
                if has_protos:
                    proto_matrix = np.concatenate(proto_feats, axis=0)
                    X_total = np.concatenate([sample_feats, proto_matrix], axis=0)
                else:
                    X_total = sample_feats

                n_total = X_total.shape[0]
                if n_total < 5:
                    print(f"[Client {self.id}] Cảnh báo: Số lượng điểm n_total = {n_total} < 5, không đủ fit t-SNE.")
                    return None

                perplexity = min(30, max(2, (n_total - 1) // 3))
                X_emb = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=0).fit_transform(X_total)

                fig, ax = plt.subplots(figsize=(8, 6))
                cmap_name = 'tab10' if n_classes <= 10 else ('tab20' if n_classes <= 20 else 'gist_ncar')
                sc = ax.scatter(X_emb[:n_samples, 0], X_emb[:n_samples, 1], c=sample_labels, cmap=cmap_name, alpha=0.6, s=20)
                plt.colorbar(sc, ax=ax, label='Class Label')

                if has_protos:
                    proto_emb = X_emb[n_samples:]
                    for idx_p, c_id in enumerate(proto_classes):
                        px, py = proto_emb[idx_p, 0], proto_emb[idx_p, 1]
                        ax.scatter(px, py, marker='*', color='black', edgecolor='red', s=220, linewidth=1.5, zorder=5)
                        ax.text(px, py, f" P_{c_id}", fontsize=9, fontweight='bold', color='black', zorder=6)

                prev_round = max(0, round_idx - 1)
                if has_protos:
                    proto_info = f"prototype gửi ở round {prev_round}"
                else:
                    proto_info = "chưa có prototype"
                sil_text = f" (Sil: {sil:.3f})" if sil is not None else ""
                title = f"client {self.id}, model sau round {prev_round} (nếu được chọn), {proto_info}{sil_text}"
                ax.set_title(title)
                ax.set_xlabel("t-SNE 1")
                ax.set_ylabel("t-SNE 2")

                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"client_{self.id}_round_{round_idx}_tsne.png")
                fig.savefig(save_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                # Ghi log WandB an toàn
                if getattr(self.args, 'log', False):
                    try:
                        if wandb.run is not None:
                            log_dict = {f"tsne/client_{self.id}": wandb.Image(save_path)}
                            if sil is not None:
                                log_dict[f"tsne/client_{self.id}_silhouette"] = sil
                            wandb.log(log_dict, commit=False)
                    except Exception:
                        pass

                return save_path
        finally:
            self.model.train()


ClientNewProto = clientNewProto

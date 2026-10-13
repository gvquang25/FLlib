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

from flcore.clients.clientbase import Client


class clientNew(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # Thuộc tính FedNew gốc
        self.global_prototypes = {}
        self.global_stds = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()
        self.lamda = getattr(args, 'lamda', 1.0)

        # Bổ sung tham số FPL: Sửa lỗi dùng toán tử 'or' để hỗ trợ giá trị 0.0 khi ablation
        v_proto = getattr(args, 'lamda_proto', 1.0)
        self.lamda_proto = 1.0 if v_proto is None else float(v_proto)

        v_temp = getattr(args, 'infoNCET', 0.1)
        self.infoNCET = 0.1 if v_temp is None else float(v_temp)

        v_warmup = getattr(args, 'proto_warmup', 1)
        self.proto_warmup = 1 if v_warmup is None else int(v_warmup)

        # Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # Cấu hình Optimizer an toàn chống cập nhật đúp tham số
        enc_lr = getattr(args, 'encoder_lr', None)
        head_params = list(self.head.parameters())
        if enc_lr is None or enc_lr <= 0 or len(head_params) == 0:
            self.opt_encoder = None
            self.opt_encoder_scheduler = None
        else:
            opt_cls = type(self.optimizer)
            def_cfg = copy.deepcopy(self.optimizer.defaults)
            enc_cfg, head_cfg = copy.deepcopy(def_cfg), copy.deepcopy(def_cfg)
            enc_cfg['lr'], head_cfg['lr'] = float(enc_lr), self.learning_rate

            self.opt_encoder = opt_cls(self.encoder.parameters(), **enc_cfg)
            self.optimizer = opt_cls(self.head.parameters(), **head_cfg)

            assert set(self.encoder.parameters()).isdisjoint(set(self.head.parameters())), "Params không rời nhau!"
            assert (set(self.encoder.parameters()) | set(self.head.parameters())) == set(self.model.parameters())

            if self.learning_rate_decay:
                sched_cls = type(self.learning_rate_scheduler)
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.learning_rate_scheduler = sched_cls(self.optimizer, gamma=gamma)
                self.opt_encoder_scheduler = sched_cls(self.opt_encoder, gamma=gamma)
            else:
                self.opt_encoder_scheduler = None

    def _split_model(self):
        """Tách backbone (trích xuất đặc trưng) và head phân lớp."""
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            return self.model.base, self.model.head
        if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            return self.model.features, self.model.classifier
        return self.model, nn.Identity()

    def extract_features(self, x):
        """
        Trích xuất đặc trưng và logits:
        - Xử lý đầu vào x (nếu là list ảnh như một số bộ dữ liệu phân đoạn).
        - rep: vector đặc trưng 2D (B, D).
        - out: logits phân lớp. Lưu ý: một số head trong PFLlib nhận tensor đa chiều (B, C, H, W)
          trong khi head dạng Linear nhận tensor 2D phẳng (B, D). Do đó dùng fallback an toàn.
        """
        x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        rep_raw = self.encoder(x)
        rep = rep_raw.view(rep_raw.size(0), -1)
        try:
            out = self.head(rep_raw)
        except (RuntimeError, ValueError):
            out = self.head(rep)
        return rep, out

    def _build_proto_cache(self):
        """
        Tiền xử lý Prototype Cache trên GPU:
        - Bổ sung proto_warmup để tránh học sai từ prototype ngẫu nhiên ở Round 0.
        - Dựng Lookup Table (LUT) trên GPU để tra cứu target_indices trong O(1) mà không gọi .item().
        """
        warmup_rounds = getattr(self.args, 'proto_warmup', 1)
        if self.train_time_cost['num_rounds'] < warmup_rounds:
            return None

        if not self.global_prototypes:
            return None

        valid_classes = sorted([
            int(k) for k, v in self.global_prototypes.items()
            if v is not None and torch.norm(v.float()) > 1e-8
        ])
        if len(valid_classes) < 2:
            return None

        # Ma trận Prototype kích thước (K, D) đã detach trên GPU
        proto_tensor = torch.stack([self.global_prototypes[c].detach().float().view(-1) for c in valid_classes]).to(self.device)

        # Xây dựng Lookup Table trên GPU
        max_cls = max(valid_classes)
        num_classes_attr = getattr(self, 'num_classes', 0)
        table_size = max(num_classes_attr, max_cls + 1)
        lut = torch.full((table_size,), -1, dtype=torch.long, device=self.device)
        valid_cls_tensor = torch.tensor(valid_classes, dtype=torch.long, device=self.device)
        indices_tensor = torch.arange(len(valid_classes), dtype=torch.long, device=self.device)
        lut[valid_cls_tensor] = indices_tensor

        return {
            'lut': lut,
            'proto_matrix': proto_tensor
        }

    def compute_fpl_loss(self, rep, y, cache):
        """
        Tính toán song song Loss FPL (InfoNCE + MSE) cho cả batch, giữ nguyên 100% công thức toán học:
        - Không đồng bộ CPU-GPU (không dùng .item() hay vòng lặp).
        - Chia mẫu hợp lệ cho tổng số mẫu batch len(y) chuẩn FPL.
        """
        # Tra cứu nhãn qua GPU LUT
        # Đảm bảo nhãn không vượt ngoài giới hạn bảng tra
        valid_label_mask = (y >= 0) & (y < cache['lut'].size(0))
        target_indices = torch.full_like(y, -1, dtype=torch.long)
        target_indices[valid_label_mask] = cache['lut'][y[valid_label_mask]]

        mask = target_indices >= 0
        if not mask.any():
            return None

        sub_rep = rep[mask]
        sub_targets = target_indices[mask]
        protos = cache['proto_matrix']  # (K, D)

        # 1. InfoNCE song song (Cosine Similarity qua ma trận tích vô hướng)
        f_norm = F.normalize(sub_rep, dim=1)
        p_norm = F.normalize(protos, dim=1)
        sim_matrix = torch.matmul(f_norm, p_norm.T) / self.infoNCET  # (B_v, K)
        loss_infonce = F.cross_entropy(sim_matrix, sub_targets, reduction='sum')

        # 2. MSE Loss với mean prototype tương ứng
        pos_protos = protos[sub_targets]  # (B_v, D)
        loss_mse = F.mse_loss(sub_rep, pos_protos, reduction='sum') / sub_rep.size(1)

        # Chuẩn FPL: cộng dồn các mẫu hợp lệ rồi chia cho tổng kích thước batch B
        return (loss_infonce + loss_mse) / len(y)

    def extract_local_stats(self, dataloader=None):
        """
        Trích xuất mean, std, count theo từng class:
        - Chỉ forward qua encoder, không qua head để tối ưu tốc độ.
        - Xử lý biên: 1 mẫu (std = 1e-6), 0 mẫu (mean/std = 0).
        """
        dataloader = dataloader or self.load_train_data()
        self.model.eval()
        features_per_class = defaultdict(list)
        feat_dim = None

        with torch.no_grad():
            for x, y in dataloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Chỉ forward qua encoder để tiết kiệm FLOPs
                rep_raw = self.encoder(x)
                rep = rep_raw.view(rep_raw.size(0), -1)

                if feat_dim is None and rep.size(0) > 0:
                    feat_dim = rep.size(1)
                assert rep.size(1) == feat_dim, f"Dimension mismatch: {rep.size(1)} vs {feat_dim}"

                for feat, label in zip(rep, y):
                    features_per_class[int(label.item())].append(feat.unsqueeze(0))

        feat_dim = feat_dim or 512
        num_classes = getattr(self, 'num_classes', 10)
        all_classes = set(range(num_classes)).union(features_per_class.keys())

        client_stats = {}
        for c in sorted(all_classes):
            c_int = int(c)
            if c_int in features_per_class and len(features_per_class[c_int]) > 0:
                class_feats = torch.cat(features_per_class[c_int], dim=0)
                n_samples = class_feats.size(0)
                mean_c = torch.mean(class_feats, dim=0)
                std_c = torch.std(class_feats, dim=0, unbiased=False) if n_samples > 1 else torch.full_like(mean_c, 1e-6)
                client_stats[c_int] = {'mean': mean_c.cpu(), 'std': std_c.cpu(), 'count': int(n_samples)}
            else:
                client_stats[c_int] = {'mean': torch.zeros(feat_dim), 'std': torch.zeros(feat_dim), 'count': 0}

        self.local_stats = client_stats
        return client_stats

    def receive_global_stats(self, global_stats):
        """Cập nhật dict mà không làm mất các prototype đã tồn tại trước đó."""
        if not global_stats:
            return
        for c, stats in global_stats.items():
            if stats and isinstance(stats, dict):
                k = int(c)
                if stats.get('mean') is not None:
                    self.global_prototypes[k] = stats['mean'].clone().detach().to(self.device)
                if stats.get('std') is not None:
                    self.global_stds[k] = stats['std'].clone().detach().to(self.device)

    def set_protos(self, global_stats):
        self.receive_global_stats(global_stats)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # Bỏ qua prototype loss nếu đang trong giai đoạn warmup hoặc lamda_proto == 0.0
        if self.train_time_cost['num_rounds'] < self.proto_warmup or self.lamda_proto == 0.0:
            cache = None
        else:
            cache = self._build_proto_cache()
        max_local_steps = np.random.randint(1, max(2, self.local_epochs // 2 + 1)) if self.train_slow else self.local_epochs

        ce_losses, total_losses, proto_losses = [], [], []

        for _ in range(max_local_steps):
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep, out = self.extract_features(x)
                loss_ce = self.loss(out, y)

                loss_InfoNCE = self.compute_fpl_loss(rep, y, cache) if cache else None
                if loss_InfoNCE is None:
                    loss_InfoNCE = 0.0 * loss_ce

                loss_total = loss_ce + self.lamda_proto * loss_InfoNCE

                self.optimizer.zero_grad()
                if self.opt_encoder:
                    self.opt_encoder.zero_grad()
                loss_total.backward()
                self.optimizer.step()
                if self.opt_encoder:
                    self.opt_encoder.step()

                # Đồng nhất: ce_losses dùng để đại diện cho train_loss theo chuẩn FedNew baseline
                ce_losses.append(loss_ce.item())
                total_losses.append(loss_total.item())
                proto_losses.append(loss_InfoNCE.item())

        # Ghi log WandB an toàn
        if getattr(self.args, 'log', False):
            try:
                if wandb.run is not None:
                    wandb.log({
                        f"clients/client_{self.id}_train_loss": float(np.mean(ce_losses)),  # CE loss chuẩn baseline
                        f"clients/client_{self.id}_loss_total": float(np.mean(total_losses)),
                        f"clients/client_{self.id}_loss_ce": float(np.mean(ce_losses)),
                        f"clients/client_{self.id}_loss_infonce": float(np.mean(proto_losses))
                    }, commit=False)
            except Exception:
                pass

        self.extract_local_stats(trainloader)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()
            if self.opt_encoder_scheduler:
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


ClientNew = clientNew
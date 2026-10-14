from collections import defaultdict
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientbase import Client
from utils.tsne_utils import plot_client_tsne
from utils.wandb_utils import log_client_train


class clientNewProto(Client):
    """
    Client cho FedNewProto:
    Kế thừa trực tiếp từ Client (clientbase), hoạt động độc lập với clientNew.
    Tích hợp Prototypical Contrastive Loss theo cách tính của FPL
    (kết hợp InfoNCE và MSE tới global prototype của từng class).
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # 1.0 Các thuộc tính lưu trữ Prototype và thống kê
        self.global_prototypes = {}
        self.global_stds = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()
        self.lamda = getattr(args, 'lamda', 1.0)

        # 1.1 Tham số nhiệt độ InfoNCE và hệ số điều chuẩn Prototypical Contrastive Loss
        v_temp = getattr(args, 'infoNCET', 0.1)
        self.infoNCET = 0.1 if v_temp is None else float(v_temp)

        v_proto = getattr(args, 'lamda_proto', 1.0)
        self.lamda_proto = 1.0 if v_proto is None else float(v_proto)

        v_warmup = getattr(args, 'proto_warmup', 1)
        self.proto_warmup = 1 if v_warmup is None else int(v_warmup)

        # 1.2 Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # 1.3 Cấu hình Optimizer an toàn
        encoder_lr = getattr(args, 'encoder_lr', None)
        head_params = list(self.head.parameters())

        if encoder_lr is None or encoder_lr <= 0 or len(head_params) == 0:
            # Trường hợp mặc định: giữ nguyên self.optimizer gốc cho toàn model
            self.opt_encoder = None
            self.opt_encoder_scheduler = None

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

    def train(self):
        """
        Huấn luyện cục bộ:
        - CE Loss trên logits phân lớp
        - Prototypical Contrastive Loss (InfoNCE + MSE) tính song song trên feature representations
        - Cập nhật thống kê cục bộ (mean, std, count) sau epoch
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        # Bỏ qua prototype loss nếu đang trong giai đoạn warmup hoặc lamda_proto == 0.0
        if self.train_time_cost['num_rounds'] < getattr(self, 'proto_warmup', 1) or self.lamda_proto == 0.0:
            cache = None
        else:
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

                loss_InfoNCE = self.compute_fpl_loss(rep, y, cache) if cache else None
                if loss_InfoNCE is None:
                    loss_InfoNCE = 0.0 * loss_ce

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
        log_client_train(self, ce_losses, extra_losses={"loss_infonce": proto_losses})

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

    def visualize_tsne(self, round_idx=0, max_samples=1000, save_dir="./tsne_resultsnewproto"):
        return plot_client_tsne(self, round_idx=round_idx, max_samples=max_samples, save_dir=save_dir)

ClientNewProto = clientNewProto
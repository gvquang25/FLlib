
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
    Client cho FedNewProto 
    - Kế thừa trực tiếp từ Client (clientbase), tách biệt hoàn toàn Encoder và Head.
    - Tận dụng đầy đủ cả Mean (global_prototypes) và Pooled Std (global_stds) từ Server:
      1. Centered InfoNCE Loss: chống co cụm góc dương của ReLU, đẩy Silhouette sang dương.
      2. Diagonal Mahalanobis Loss: kéo đặc trưng có trọng số bất định dựa trên global_stds.
      3. Gaussian Virtual Feature Calibration: sinh mẫu ảo N(mean, std^2) giúp Head chống
         overfitting trên Local Test và nhận diện được các lớp khuyết trên Global Test.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # 1.0 Các thuộc tính lưu trữ Prototype và thống kê
        self.global_prototypes = {}
        self.global_stds = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()

        # Đồng bộ chính xác tham số -lam (args.lamda) từ CLI và args.lamda_proto
        raw_lamda = getattr(args, 'lamda', None)
        self.lamda = float(raw_lamda) if raw_lamda is not None else 0.1

        v_proto = getattr(args, 'lamda_proto', None)
        if v_proto is not None and float(v_proto) != 1.0:
            self.lamda_proto = float(v_proto)
        else:
            self.lamda_proto = self.lamda

        # 1.1 Tham số nhiệt độ InfoNCE, Warmup và hệ số Virtual Feature cho Head
        v_temp = getattr(args, 'infoNCET', 0.1)
        self.infoNCET = 0.1 if v_temp is None else float(v_temp)

        v_warmup = getattr(args, 'proto_warmup', 1)
        self.proto_warmup = 1 if v_warmup is None else int(v_warmup)

        v_rampup = getattr(args, 'proto_rampup', 5)
        self.proto_rampup = 5 if v_rampup is None else max(1, int(v_rampup))

        # Hệ số điều chuẩn đặc trưng ảo Gauss cho Classifier Head
        v_virt_local = getattr(args, 'lamda_virt_local', 0.15)
        self.lamda_virt_local = 0.15 if v_virt_local is None else float(v_virt_local)

        v_virt_global = getattr(args, 'lamda_virt_global', 0.05)
        self.lamda_virt_global = 0.05 if v_virt_global is None else float(v_virt_global)

        # 1.2 Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # 1.3 Cấu hình Optimizer tách riêng cho Encoder và Head
        opt_cls = type(self.optimizer)
        def_cfg = copy.deepcopy(self.optimizer.defaults)

        head_params = list(self.head.parameters())
        enc_params = list(self.encoder.parameters())

        encoder_lr = getattr(args, 'encoder_lr', None)
        enc_lr = float(encoder_lr) if (encoder_lr is not None and float(encoder_lr) > 0) else self.learning_rate

        head_lr = getattr(args, 'head_lr', None)
        h_lr = float(head_lr) if (head_lr is not None and float(head_lr) > 0) else self.learning_rate

        if len(head_params) > 0 and len(enc_params) > 0:
            enc_cfg = copy.deepcopy(def_cfg)
            head_cfg = copy.deepcopy(def_cfg)
            enc_cfg['lr'] = enc_lr
            head_cfg['lr'] = h_lr

            self.opt_encoder = opt_cls(enc_params, **enc_cfg)
            self.opt_head = opt_cls(head_params, **head_cfg)
            self.optimizer = self.opt_head  # Giữ self.optimizer đồng bộ với opt_head cho Base Client
            self.opt_base = self.opt_encoder

            opt_enc_params = set(p for group in self.opt_encoder.param_groups for p in group['params'])
            opt_head_params = set(p for group in self.opt_head.param_groups for p in group['params'])
            assert opt_enc_params.isdisjoint(opt_head_params), "Tham số của 2 optimizer không được trùng lặp!"
            assert (opt_enc_params | opt_head_params) == set(self.model.parameters()), "Hợp tham số 2 optimizer phải bằng toàn bộ tham số model!"

            if self.learning_rate_decay:
                sched_cls = type(self.learning_rate_scheduler) if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None else torch.optim.lr_scheduler.ExponentialLR
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.learning_rate_scheduler = sched_cls(self.opt_head, gamma=gamma)
                self.opt_head_scheduler = self.learning_rate_scheduler
                self.opt_encoder_scheduler = sched_cls(self.opt_encoder, gamma=gamma)
            else:
                self.opt_encoder_scheduler = None
                self.opt_head_scheduler = None
        elif len(head_params) == 0:
            # Fallback nếu head không có trainable params (nn.Identity)
            enc_cfg = copy.deepcopy(def_cfg)
            enc_cfg['lr'] = enc_lr
            self.opt_encoder = opt_cls(enc_params, **enc_cfg)
            self.opt_head = None
            self.optimizer = self.opt_encoder
            self.opt_base = self.opt_encoder
            if self.learning_rate_decay:
                sched_cls = type(self.learning_rate_scheduler) if hasattr(self, 'learning_rate_scheduler') and self.learning_rate_scheduler is not None else torch.optim.lr_scheduler.ExponentialLR
                gamma = getattr(self.args, 'learning_rate_decay_gamma', 0.99)
                self.learning_rate_scheduler = sched_cls(self.opt_encoder, gamma=gamma)
                self.opt_encoder_scheduler = self.learning_rate_scheduler
                self.opt_head_scheduler = None
            else:
                self.opt_encoder_scheduler = None
                self.opt_head_scheduler = None

    def _split_model(self):
        """
        Tách backbone (encoder trích xuất đặc trưng) và head phân lớp:
        - Ưu tiên 1: model.base / model.head (BaseHeadSplit)
        - Ưu tiên 2: model.features / model.classifier
        - Fallback: encoder = model, head = nn.Identity()
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
        Trích xuất biểu diễn đặc trưng rep và logits out.
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
        dataloader = dataloader or self.load_train_data()
        self.model.eval()
        features_per_class = defaultdict(list)
        feat_dim = None

        with torch.no_grad():
            for x, y in dataloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

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
        """Cập nhật dict mean và std toàn cục mà không làm mất các prototype đã tồn tại trước đó."""
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
        warmup_rounds = getattr(self, 'proto_warmup', 1)
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

        proto_list = []
        std_list = []
        for c in valid_classes:
            p_vec = self.global_prototypes[c].detach().float().view(-1).to(self.device)
            proto_list.append(p_vec)

            if c in self.global_stds and self.global_stds[c] is not None:
                s_vec = self.global_stds[c].detach().float().view(-1).to(self.device)
                if torch.norm(s_vec) <= 1e-6:
                    s_vec = torch.full_like(p_vec, 0.1)
            else:
                s_vec = torch.full_like(p_vec, 0.1)
            std_list.append(s_vec)

        proto_tensor = torch.stack(proto_list, dim=0)  # (K, D)
        std_tensor = torch.clamp(torch.stack(std_list, dim=0), min=1e-4)  # (K, D)

        # Tâm toàn cục của các prototype giúp giải quyết hiện tượng co cụm góc dương của ReLU
        proto_center = torch.mean(proto_tensor, dim=0, keepdim=True)  # (1, D)

        # Tính mẫu số phương sai an toàn cho khoảng cách Mahalanobis đường chéo
        var_tensor = std_tensor.pow(2)
        mean_var = float(torch.mean(var_tensor).item())
        tau_var = max(0.25 * mean_var, 1e-2)
        mahalanobis_denom = var_tensor + tau_var  # (K, D)

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
            'valid_classes': valid_cls_tensor,
            'proto_matrix': proto_tensor,
            'std_matrix': std_tensor,
            'proto_center': proto_center,
            'mahalanobis_denom': mahalanobis_denom
        }

    def compute_fpl_loss(self, rep, y, cache):
        """
        Tính toán song song Prototypical Contrastive + Uncertainty-Aware Mahalanobis Loss:
        1. Centered InfoNCE: Đẩy các lớp khác nhau và kéo lớp mục tiêu trên mặt cầu tâm hóa.
        2. Diagonal Mahalanobis Loss: Căn chỉnh khoảng cách Euclidean có trọng số phương sai gộp (global_stds).
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
        center = cache['proto_center']  # (1, D)

        # 1. Centered InfoNCE: Tâm hóa đặc trưng quanh proto_center trước khi chuẩn hóa L2
        # Giúp các vector sau ReLU có thể phân tách góc lên tới [-1, 1] thay vì bị kẹt ở [0, 1]
        f_centered = sub_rep - center
        p_centered = protos - center
        f_norm = F.normalize(f_centered, dim=1)
        p_norm = F.normalize(p_centered, dim=1)
        sim_matrix = torch.matmul(f_norm, p_norm.T) / self.infoNCET  # (B_v, K)
        loss_infonce = F.cross_entropy(sim_matrix, sub_targets, reduction='sum')

        # 2. Diagonal Mahalanobis Alignment (Tận dụng trực tiếp Pooled Variance từ Server)
        # Chiều nào có std nhỏ (độ tin cậy cao) -> lực kéo mạnh; std lớn -> nới lỏng tránh xung đột
        pos_protos = protos[sub_targets]  # (B_v, D)
        pos_denom = cache['mahalanobis_denom'][sub_targets]  # (B_v, D)
        mahalanobis_dist = (sub_rep - pos_protos).pow(2) / pos_denom
        loss_mahalanobis = torch.sum(mahalanobis_dist) / sub_rep.size(1)

        return (loss_infonce + loss_mahalanobis) / len(y)

    def _compute_virtual_head_loss(self, y, cache):
        """
        Sinh đặc trưng ảo Gauss từ (global_prototypes, global_stds) để hiệu chỉnh Classifier Head:
        - Nhánh 1 (Local-Prior Virtual): Lấy mẫu theo đúng nhãn y của batch hiện tại để chống
          overfitting và nâng độ chính xác cho các lớp thiểu số trên tập Local Test.
        - Nhánh 2 (Global-Class Virtual): Lấy mẫu trên toàn bộ K lớp toàn cục để giữ ranh giới
          phân loại cho các lớp bị khuyết, cải thiện trực tiếp Global Test Accuracy.
        """
        if cache is None or self.opt_head is None:
            return None

        protos = cache['proto_matrix']       # (K, D)
        stds = cache['std_matrix']           # (K, D)
        valid_classes = cache['valid_classes']  # (K,)

        total_virt_loss = None

        # Nhánh 1: Giữ nguyên phân phối nhãn cục bộ y của client
        if self.lamda_virt_local > 0:
            valid_label_mask = (y >= 0) & (y < cache['lut'].size(0))
            target_indices = torch.full_like(y, -1, dtype=torch.long)
            target_indices[valid_label_mask] = cache['lut'][y[valid_label_mask]]
            mask = target_indices >= 0

            if mask.any():
                sub_targets = target_indices[mask]
                y_valid = y[mask]
                pos_protos = protos[sub_targets]
                pos_stds = stds[sub_targets]

                eps_local = torch.randn_like(pos_protos)
                z_virt_local = (pos_protos + 0.5 * eps_local * pos_stds).detach()
                try:
                    out_virt_local = self.head(z_virt_local)
                    loss_v_local = self.loss(out_virt_local, y_valid)
                    total_virt_loss = self.lamda_virt_local * loss_v_local
                except (RuntimeError, ValueError):
                    pass

        # Nhánh 2: Hiệu chỉnh nhẹ toàn bộ K lớp cho Global Test
        if self.lamda_virt_global > 0:
            eps_all = torch.randn_like(protos)
            z_virt_all = (protos + 0.3 * eps_all * stds).detach()
            try:
                out_virt_all = self.head(z_virt_all)
                loss_v_global = self.loss(out_virt_all, valid_classes)
                virt_term = self.lamda_virt_global * loss_v_global
                total_virt_loss = virt_term if total_virt_loss is None else (total_virt_loss + virt_term)
            except (RuntimeError, ValueError):
                pass

        return total_virt_loss

    def train(self):
        """
        Huấn luyện cục bộ với kiến trúc tách rời (Decoupled Encoder & Head):
        - Head: cập nhật bằng CE Loss trên feats.detach() + Gaussian Virtual Features từ (mean, std).
        - Encoder: cập nhật bằng CE Loss + Progressive Prototype Loss (Centered InfoNCE + Mahalanobis).
        """
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        num_rounds = self.train_time_cost['num_rounds']
        warmup_rounds = getattr(self, 'proto_warmup', 1)

        if num_rounds < warmup_rounds or self.lamda_proto == 0.0:
            cache = None
            eff_lamda_proto = 0.0
        else:
            cache = self._build_proto_cache()
            # Ramp-up mượt hệ số lamda_proto trong những vòng đầu sau warmup để tránh sụp đổ sớm
            ramp_progress = min(1.0, float(num_rounds - warmup_rounds + 1) / float(self.proto_rampup))
            eff_lamda_proto = self.lamda_proto * ramp_progress

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

                x_input = x[0] if isinstance(x, list) else x

                # Trích xuất đặc trưng từ Encoder
                feats = self.encoder(x_input)
                rep = feats.view(feats.size(0), -1)

                # 1. Cập nhật Classifier Head (ngắt gradient từ Encoder + bổ sung đặc trưng ảo Gauss)
                try:
                    output = self.head(feats.detach())
                except (RuntimeError, ValueError):
                    output = self.head(rep.detach())
                loss_ce = self.loss(output, y)

                loss_virt_head = self._compute_virtual_head_loss(y, cache) if cache else None
                loss_head = loss_ce + loss_virt_head if loss_virt_head is not None else loss_ce

                # 2. Forward Head giữ computation graph để Encoder nhận tín hiệu phân loại L_ce
                try:
                    output_enc = self.head(feats)
                except (RuntimeError, ValueError):
                    output_enc = self.head(rep)
                loss_ce_enc = self.loss(output_enc, y)

                # 3. Tính Prototype Loss (Centered InfoNCE + Mahalanobis dùng global_stds) cho Encoder
                loss_InfoNCE = self.compute_fpl_loss(rep, y, cache) if cache else None
                if loss_InfoNCE is None:
                    loss_InfoNCE = 0.0 * loss_ce

                loss_encoder = loss_ce_enc + eff_lamda_proto * loss_InfoNCE

                # Zero gradients cho cả 2 optimizer
                if self.opt_head is not None:
                    self.opt_head.zero_grad()
                if self.opt_encoder is not None:
                    self.opt_encoder.zero_grad()

                # Backward cho Head
                if self.opt_head is not None:
                    loss_head.backward()

                # Backward cô lập chỉ truyền vào tham số của Encoder
                if self.opt_encoder is not None:
                    enc_params = list(self.encoder.parameters())
                    loss_encoder.backward(inputs=enc_params)

                # Step từng optimizer riêng biệt
                if self.opt_head is not None:
                    self.opt_head.step()
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
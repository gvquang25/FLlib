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


class clientNewPCL(Client):
    """
    Client cho FedNewPCL (Prototypical Contrastive Learning in Federated Learning):
    - Kế thừa trực tiếp từ Client (clientbase), hoạt động hoàn toàn độc lập.
    - Duy trì kiến trúc tách rời (Decoupled Encoder & Classifier Head) với 2 optimizer riêng biệt.
    - Tích hợp M-Step cục bộ dựa trên Prototypical Contrastive Learning (PCL):
      1. Centered ProtoNCE với Adaptive Temperature phi được ước lượng động từ Server.
      2. Intra-batch Instance Contrastive Loss (học tương phản giữa các mẫu cục bộ).
      3. Diagonal Mahalanobis Alignment (căn chỉnh theo phương sai gộp).
      4. Gaussian Virtual Feature Calibration (điều chuẩn Head bằng mẫu ảo Gauss).
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # 1.0 Thuộc tính lưu trữ Prototype, thống kê phân tán và nhiệt độ động PCL
        self.global_prototypes = {}
        self.global_stds = {}
        self.global_phis = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()

        # Đồng bộ trọng số điều chuẩn prototype
        raw_lamda = getattr(args, 'lamda', None)
        self.lamda = float(raw_lamda) if raw_lamda is not None else 0.1

        v_proto = getattr(args, 'lamda_proto', None)
        if v_proto is not None and float(v_proto) != 1.0:
            self.lamda_proto = float(v_proto)
        else:
            self.lamda_proto = self.lamda

        # 1.1 Tham số nhiệt độ cơ sở, Intra-batch Instance loss, Warmup và Ramp-up
        v_temp = getattr(args, 'infoNCET', 0.1)
        self.infoNCET = 0.1 if v_temp is None else float(v_temp)

        v_ins = getattr(args, 'lamda_ins', 0.1)
        self.lamda_ins = 0.1 if v_ins is None else float(v_ins)

        v_warmup = getattr(args, 'proto_warmup', 1)
        self.proto_warmup = 1 if v_warmup is None else int(v_warmup)

        v_rampup = getattr(args, 'proto_rampup', 5)
        self.proto_rampup = 5 if v_rampup is None else max(1, int(v_rampup))

        # Hệ số điều chuẩn đặc trưng ảo Gauss cho Classifier Head (tối ưu cho Personalized Acc)
        v_virt_local = getattr(args, 'lamda_virt_local', 0.05)
        self.lamda_virt_local = 0.05 if v_virt_local is None else float(v_virt_local)

        v_virt_global = getattr(args, 'lamda_virt_global', 0.03)
        self.lamda_virt_global = 0.03 if v_virt_global is None else float(v_virt_global)

        # 1.2 Tách Backbone (Encoder) và Classification Head
        self.encoder, self.head = self._split_model()

        # 1.3 Cấu hình 2 Optimizer tách riêng cho Encoder và Head
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
            self.optimizer = self.opt_head  # Đồng bộ self.optimizer cho Base Client
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
        """Cập nhật dict mean, std và phi toàn cục mà không làm mất các prototype đã tồn tại."""
        if not global_stats:
            return
        for c, stats in global_stats.items():
            if stats and isinstance(stats, dict):
                k = int(c)
                if stats.get('mean') is not None:
                    self.global_prototypes[k] = stats['mean'].clone().detach().to(self.device)
                if stats.get('std') is not None:
                    self.global_stds[k] = stats['std'].clone().detach().to(self.device)
                if stats.get('phi') is not None:
                    self.global_phis[k] = float(stats['phi'])

    def set_protos(self, global_stats):
        self.receive_global_stats(global_stats)

    def _build_proto_cache(self):
        """
        Chuẩn bị Cache Prototype, Pooled Variance, Mahalanobis Denominator và Dynamic Temperature phi trên GPU:
        - Bỏ qua nếu đang trong giai đoạn proto_warmup.
        - Xây dựng Lookup Table (LUT) song song trên GPU cho O(1) index mapping.
        """
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
        phi_list = []

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

            # Lấy nhiệt độ động phi của cụm c (fallback về infoNCET nếu chưa có)
            phi_val = self.global_phis.get(c, self.infoNCET)
            phi_list.append(float(phi_val))

        proto_tensor = torch.stack(proto_list, dim=0)  # (K, D)
        std_tensor = torch.clamp(torch.stack(std_list, dim=0), min=1e-4)  # (K, D)

        # Tâm toàn cục của các prototype
        proto_center = torch.mean(proto_tensor, dim=0, keepdim=True)  # (1, D)

        # Mẫu số khoảng cách Mahalanobis đường chéo ĐÃ CHUẨN HÓA TƯƠNG ĐỐI (tránh phóng đại gradient)
        var_tensor = std_tensor.pow(2)
        mean_var = float(torch.mean(var_tensor).item())
        tau_var = max(0.25 * mean_var, 1e-2)
        # Chia cho (mean_var + tau_var) để trung bình trọng số bằng 1.0, không làm nổ đạo hàm
        mahalanobis_denom = (var_tensor + tau_var) / (mean_var + tau_var)  # (K, D)

        # Vector nhiệt độ động phi (1, K) trên GPU
        phi_tensor = torch.tensor(phi_list, dtype=torch.float32, device=self.device).view(1, -1)  # (1, K)

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
            'mahalanobis_denom': mahalanobis_denom,
            'phi_tensor': phi_tensor
        }

    def compute_pcl_loss(self, rep, y, cache):
        """
        Hàm mất mát PCL hoàn chỉnh cho client cục bộ (kế thừa từ pcl/builder.py và main_pcl.py):
        1. Centered ProtoNCE với Adaptive Dynamic Temperature phi:
           - Tâm hóa rep quanh proto_center, chuẩn hóa L2: f_norm, p_norm
           - sim_matrix = (f_norm @ p_norm.T) / phi_tensor
           - loss_protonce = cross_entropy(sim_matrix, sub_targets, reduction='sum')
        2. Intra-batch Instance Contrastive (cơ chế đa cấp của PCL):
           - Tính tương phản giữa các mẫu cục bộ có cùng nhãn trên f_norm
           - Tỷ lệ theo self.lamda_ins
        3. Diagonal Mahalanobis Alignment:
           - Khoảng cách sai lệch bình phương chia cho mahalanobis_denom
        4. Tổng hợp và chuẩn hóa chia cho len(y)
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

        # 1. Centered ProtoNCE với Adaptive Dynamic Temperature phi
        f_centered = sub_rep - center
        p_centered = protos - center
        f_norm = F.normalize(f_centered, dim=1)  # (B_v, D)
        p_norm = F.normalize(p_centered, dim=1)  # (K, D)

        # Tích vô hướng Cosine chia trực tiếp cho vector nhiệt độ động (1, K) của từng cụm
        sim_matrix = torch.matmul(f_norm, p_norm.T) / cache['phi_tensor']  # (B_v, K)
        loss_protonce = F.cross_entropy(sim_matrix, sub_targets, reduction='sum')

        # 2. Intra-batch Instance Contrastive (theo cơ chế đa cấp của PCL)
        n_samples = f_norm.size(0)
        loss_instance = torch.tensor(0.0, device=self.device)
        if n_samples > 1 and self.lamda_ins > 0:
            eye = torch.eye(n_samples, dtype=torch.bool, device=self.device)
            pos_mask = (sub_targets.unsqueeze(1) == sub_targets.unsqueeze(0)) & (~eye)
            pos_counts = pos_mask.sum(dim=1)
            valid_anchors = pos_counts > 0

            if valid_anchors.any():
                sim_inst = torch.matmul(f_norm, f_norm.T) / self.infoNCET  # (N, N)
                sim_inst = sim_inst.masked_fill(eye, -1e9)
                sim_max, _ = torch.max(sim_inst, dim=1, keepdim=True)
                exp_sim = torch.exp(sim_inst - sim_max) * (~eye)
                denom = exp_sim.sum(dim=1, keepdim=True)
                log_prob = (sim_inst - sim_max) - torch.log(denom + 1e-8)
                mean_log_prob_pos = (pos_mask[valid_anchors] * log_prob[valid_anchors]).sum(dim=1) / pos_counts[valid_anchors].float()
                loss_instance = -mean_log_prob_pos.sum()

        # 3. Diagonal Mahalanobis Alignment (đã chuẩn hóa thang đo)
        pos_protos = protos[sub_targets]  # (B_v, D)
        pos_denom = cache['mahalanobis_denom'][sub_targets]  # (B_v, D)
        mahalanobis_dist = (sub_rep - pos_protos).pow(2) / pos_denom
        loss_mahalanobis = 0.5 * (torch.sum(mahalanobis_dist) / sub_rep.size(1))

        # 4. Tổng hợp và chuẩn hóa chia cho tổng kích thước batch len(y)
        total_loss = (loss_protonce + loss_mahalanobis + self.lamda_ins * loss_instance) / len(y)
        return total_loss

    def _compute_virtual_head_loss(self, y, cache):
        """
        Sinh đặc trưng ảo Gauss từ (proto_matrix, std_matrix) để hiệu chỉnh Classifier Head:
        - Nhánh 1 (Local-Prior Virtual): Lấy mẫu theo đúng nhãn y của batch hiện tại.
        - Nhánh 2 (Global-Class Virtual): Lấy mẫu trên toàn bộ K lớp toàn cục để giữ ranh giới cho lớp khuyết.
        """
        if cache is None or self.opt_head is None:
            return None

        protos = cache['proto_matrix']          # (K, D)
        stds = cache['std_matrix']              # (K, D)
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
                z_virt_local = (pos_protos + 0.2 * eps_local * pos_stds).detach()
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
        Huấn luyện cục bộ M-Step với kiến trúc tách rời (Decoupled Encoder & Head):
        - Head: cập nhật bằng CE Loss trên feats.detach() + Gaussian Virtual Features.
        - Encoder: cập nhật bằng CE Loss + Progressive PCL Loss (ProtoNCE + Instance + Mahalanobis).
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
            ramp_progress = min(1.0, float(num_rounds - warmup_rounds + 1) / float(self.proto_rampup))
            eff_lamda_proto = self.lamda_proto * ramp_progress

        max_local_steps = np.random.randint(1, max(2, self.local_epochs // 2 + 1)) if self.train_slow else self.local_epochs

        ce_losses = []
        pcl_losses = []

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

                # 1. Cập nhật Classifier Head (ngắt gradient từ Encoder + đặc trưng ảo Gauss)
                try:
                    output = self.head(feats.detach())
                except (RuntimeError, ValueError):
                    output = self.head(rep.detach())
                loss_ce = self.loss(output, y)

                loss_virt_head = self._compute_virtual_head_loss(y, cache) if cache else None
                loss_head = loss_ce + loss_virt_head if loss_virt_head is not None else loss_ce

                # 2. Forward Head giữ computation graph để Encoder nhận tín hiệu phân loại
                try:
                    output_enc = self.head(feats)
                except (RuntimeError, ValueError):
                    output_enc = self.head(rep)
                loss_ce_enc = self.loss(output_enc, y)

                # 3. Tính PCL Loss (ProtoNCE với nhiệt độ động phi + Intra-batch Instance + Mahalanobis)
                loss_pcl = self.compute_pcl_loss(rep, y, cache) if cache else None
                if loss_pcl is None:
                    loss_pcl = 0.0 * loss_ce

                loss_encoder = loss_ce_enc + eff_lamda_proto * loss_pcl

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
                pcl_losses.append(loss_pcl.item())

        # Ghi log WandB an toàn nếu cờ log được bật
        log_client_train(self, ce_losses, extra_losses={"loss_pcl": pcl_losses})

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

    def visualize_tsne(self, round_idx=0, max_samples=1000, save_dir="./tsne_resultsnewpcl"):
        return plot_client_tsne(self, round_idx=round_idx, max_samples=max_samples, save_dir=save_dir)


ClientNewPCL = clientNewPCL

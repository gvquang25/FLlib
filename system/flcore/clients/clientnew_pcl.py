from collections import defaultdict
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from flcore.clients.clientdecoupled import DecoupledClient
from utils.tsne_utils import plot_client_tsne
from utils.wandb_utils import log_client_train


class clientNewPCL(DecoupledClient):
    """
    Client cho FedNewPCL (Prototypical Contrastive Learning in Federated Learning):
    - Kế thừa trực tiếp từ DecoupledClient, quản lý tách rời Encoder & Classifier Head và 2 optimizer.
    - Duy trì kiến trúc tách rời (Decoupled Encoder & Classifier Head) với 2 optimizer riêng biệt.
    - Tích hợp M-Step cục bộ dựa trên Prototypical Contrastive Learning (PCL):
      1. Adaptive ProtoNCE với Row-wise Dynamic Temperature phi.
      2. Decoupled Intra-batch Supervised Contrastive Loss.
      3. SNR-Aware Precision-Weighted Alignment.
      4. Gaussian Virtual Feature Calibration (mặc định tắt: lamda_virt = 0.0).
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

        # Đồng bộ trọng số điều chuẩn prototype theo -lam 0.1
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

        self.proto_warmup = 1 if getattr(args, 'proto_warmup', None) is None else int(args.proto_warmup)
        self.proto_rampup = 5 if getattr(args, 'proto_rampup', None) is None else max(1, int(args.proto_rampup))
        self.lamda_ins = 0.05 if getattr(args, 'lamda_ins', None) is None else float(args.lamda_ins)

        v_virt_local = getattr(args, 'lamda_virt_local', 0.0)
        self.lamda_virt_local = 0.0 if v_virt_local is None else float(v_virt_local)

        v_virt_global = getattr(args, 'lamda_virt_global', 0.05)
        self.lamda_virt_global = 0.05 if v_virt_global is None else float(v_virt_global)

        self.proj_mode = getattr(args, 'proj_mode', 'shared')
        self.use_pcl_aug = bool(getattr(args, 'pcl_aug', True))
        self.mixup_alpha = float(getattr(args, 'mixup_alpha', 0.2))
        self.bayes_shrink_max = float(getattr(args, 'bayes_shrink', 0.25))

    def _apply_pcl_view(self, x):
        """Biến đổi góc nhìn tương phản trên GPU (Random Flip + Reflection Pad Crop) chống memorization."""
        if not self.use_pcl_aug or not getattr(self.model, 'training', True) or not isinstance(x, torch.Tensor) or x.dim() != 4 or x.size(2) < 16:
            return x
        B, C, H, W = x.shape
        # 1. Lật ngang ngẫu nhiên với xác suất 0.5 cho từng ảnh trong batch
        flip_mask = (torch.rand(B, 1, 1, 1, device=x.device) < 0.5)
        x_aug = torch.where(flip_mask, torch.flip(x, dims=[3]), x)
        # 2. Dịch chuyển ngẫu nhiên (Random Crop sau khi pad phản xạ 4 pixel)
        pad = 4
        x_pad = F.pad(x_aug, (pad, pad, pad, pad), mode='reflect')
        top = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
        left = int(torch.randint(0, 2 * pad + 1, (1,), device=x.device).item())
        return x_pad[:, :, top:top + H, left:left + W]

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
        Chuẩn bị Cache Prototype, Centered Hypersphere Projection, SNR Gate và Dynamic Temperature phi trên GPU:
        - Bỏ qua nếu đang trong giai đoạn proto_warmup.
        - Áp dụng Empirical Bayes Shrinkage giữa global prototype và local prototype.
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
            g_vec = self.global_prototypes[c].detach().float().view(-1).to(self.device)
            if (
                self.bayes_shrink_max > 0
                and isinstance(self.local_stats, dict)
                and c in self.local_stats
                and self.local_stats[c].get('count', 0) > 0
            ):
                n_c = float(self.local_stats[c]['count'])
                l_vec = self.local_stats[c]['mean'].detach().float().view(-1).to(self.device)
                gamma_c = self.bayes_shrink_max * (n_c / (n_c + 50.0))
                p_vec = (1.0 - gamma_c) * g_vec + gamma_c * l_vec
            else:
                p_vec = g_vec
            proto_list.append(p_vec)

            if c in self.global_stds and self.global_stds[c] is not None:
                s_vec = self.global_stds[c].detach().float().view(-1).to(self.device)
            else:
                s_vec = torch.zeros_like(p_vec)
            std_list.append(s_vec)

            # Lấy nhiệt độ động phi của cụm c (fallback về infoNCET nếu chưa có)
            phi_val = self.global_phis.get(c, self.infoNCET)
            phi_list.append(float(phi_val))

        proto_tensor = torch.stack(proto_list, dim=0)                     # (K, D)
        std_tensor = torch.clamp(torch.stack(std_list, dim=0), min=1e-4)  # (K, D)

        # Tâm cụm toàn cục giải quyết hiện tượng co cụm nón dương sau ReLU
        proto_center = torch.mean(proto_tensor, dim=0, keepdim=True)      # (1, D)

        # Mẫu số phương sai gộp chuẩn giữ nguyên độ lợi đạo hàm ~40x khi dùng -lam 0.1
        var_tensor = std_tensor.pow(2)                                    # (K, D)
        mean_var = float(torch.mean(var_tensor).item())
        tau_var = max(0.25 * mean_var, 1e-2)
        mahalanobis_denom = var_tensor + tau_var                          # (K, D)

        # Cổng lọc tỷ số Tín hiệu (SNR Gate): giảm trọng số ở các chiều Dead ReLU (proto=0) và tăng ở chiều đặc trưng mạnh
        proto_sq = proto_tensor.pow(2)                                    # (K, D)
        mean_proto_sq = torch.mean(proto_sq, dim=1, keepdim=True) + 1e-6  # (K, 1)
        norm_signal = proto_sq / mean_proto_sq                            # (K, D)
        raw_gate = (0.5 + 0.5 * norm_signal) / (1.0 + 0.5 * norm_signal)  # (K, D)
        snr_gate = raw_gate / (torch.mean(raw_gate, dim=1, keepdim=True) + 1e-8)  # (K, D), mean(dim=1) == 1.0

        # Trọng số chiếu mặt cầu đơn vị (hỗ trợ cả shared và class)
        raw_inv_std = (1.0 / torch.sqrt(mahalanobis_denom)) * snr_gate    # (K, D)
        class_inv_std = raw_inv_std / (torch.mean(raw_inv_std, dim=1, keepdim=True) + 1e-8)
        shared_inv_std = class_inv_std.mean(dim=0, keepdim=True)
        shared_inv_std = shared_inv_std / (shared_inv_std.mean() + 1e-8)  # (1, D)

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
            'snr_gate': snr_gate,
            'class_inv_std': class_inv_std,
            'shared_inv_std': shared_inv_std,
            'phi_tensor': phi_tensor
        }

    def compute_pcl_loss(self, rep, y, cache):
        """
        Hàm mất mát PCL hoàn chỉnh cho client cục bộ trên không gian Chiếu Mặt cầu Tâm hóa:
        1. Centered Shared-Hypersphere Projection với proto_center và inv_std (shared hoặc class)
        2. Cấp độ 1 — Variance-Aware ProtoNCE với Row-wise Dynamic Temperature phi_anchor
        3. Cấp độ 2 — Decoupled Hybrid Supervised Contrastive (ghép all_feats = [f_norm, p_norm])
        4. Cấp độ 3 — SNR-Gated Mahalanobis Compactness (giữ nguyên độ lợi đạo hàm, lọc nơ-ron Dead ReLU)
        5. Tổng hợp chia cho len(y)
        """
        valid_label_mask = (y >= 0) & (y < cache['lut'].size(0))
        target_indices = torch.full_like(y, -1, dtype=torch.long)
        target_indices[valid_label_mask] = cache['lut'][y[valid_label_mask]]

        mask = target_indices >= 0
        if not mask.any():
            return None

        sub_rep = rep[mask]
        sub_targets = target_indices[mask]
        protos = cache['proto_matrix']   # (K, D)
        center = cache['proto_center']   # (1, D)

        if self.proj_mode == 'shared':
            w_s = cache['shared_inv_std']
            w_p = cache['shared_inv_std']
        else:
            w_s = cache['class_inv_std'][sub_targets]
            w_p = cache['class_inv_std']

        # 1. Chiếu đặc trưng tâm hóa lên mặt cầu đơn vị
        f_centered = (sub_rep - center) * w_s
        p_centered = (protos - center) * w_p
        f_norm = F.normalize(f_centered, dim=1)
        p_norm = F.normalize(p_centered, dim=1)

        # 2. Cấp độ 1 — Variance-Aware ProtoNCE chuẩn PCL (chia cho nhiệt độ động phi_k của từng cụm k)
        sim_proto = torch.matmul(f_norm, p_norm.T) / cache['phi_tensor']  # (B_v, K)
        loss_protonce = F.cross_entropy(sim_proto, sub_targets, reduction='sum')

        # 3. Cấp độ 2 — Decoupled Hybrid Supervised Contrastive
        loss_supcon = torch.tensor(0.0, device=self.device)
        if self.lamda_ins > 0:
            K = protos.size(0)
            all_feats = torch.cat([f_norm, p_norm], dim=0)  # (B_v + K, D)
            proto_labels = torch.arange(K, dtype=torch.long, device=self.device)
            all_targets = torch.cat([sub_targets, proto_labels], dim=0)  # (B_v + K,)
            N = all_feats.size(0)
            if N > 1:
                eye = torch.eye(N, dtype=torch.bool, device=self.device)
                pos_mask = (all_targets.unsqueeze(1) == all_targets.unsqueeze(0)) & (~eye)
                neg_mask = (all_targets.unsqueeze(1) != all_targets.unsqueeze(0))
                pos_counts = pos_mask.sum(dim=1)

                # Chỉ tính trên các anchor thuộc batch mẫu thực tế (0 <= idx < B_v)
                anchor_mask = torch.zeros(N, dtype=torch.bool, device=self.device)
                anchor_mask[:sub_rep.size(0)] = True
                valid_anchors = (pos_counts > 0) & anchor_mask

                if valid_anchors.any():
                    sim_all = torch.matmul(all_feats, all_feats.T) / self.infoNCET
                    sim_all = sim_all.masked_fill(eye, -1e9)
                    sim_max, _ = torch.max(sim_all, dim=1, keepdim=True)
                    exp_sim = torch.exp(sim_all - sim_max) * (~eye)
                    neg_exp_sum = (exp_sim * neg_mask).sum(dim=1, keepdim=True)
                    denom = exp_sim + neg_exp_sum
                    log_prob = (sim_all - sim_max) - torch.log(denom + 1e-8)
                    mean_log_prob_pos = (pos_mask[valid_anchors] * log_prob[valid_anchors]).sum(dim=1) / pos_counts[valid_anchors].float()
                    loss_supcon = -mean_log_prob_pos.sum()

        # 4. Cấp độ 3 — SNR-Gated Mahalanobis Compactness
        pos_protos = protos[sub_targets]
        pos_denom = cache['mahalanobis_denom'][sub_targets]
        pos_gate = cache['snr_gate'][sub_targets]
        loss_compact = torch.sum(pos_gate * (sub_rep - pos_protos).pow(2) / pos_denom) / sub_rep.size(1)

        # 5. Tổng hợp
        return (loss_protonce + loss_compact + self.lamda_ins * loss_supcon) / len(y)

    def _compute_virtual_head_loss(self, y, cache):
        """
        Sinh đặc trưng ảo Gauss từ (proto_matrix, std_matrix) để hiệu chỉnh Classifier Head:
        - Nhánh 1 (Local-Prior Virtual): Lấy mẫu theo đúng nhãn y của batch hiện tại.
        - Nhánh 2 (Global-Class Virtual): Lấy mẫu trên toàn bộ K lớp toàn cục để giữ ranh giới cho lớp khuyết.
        - Chỉ kẹp .clamp(min=0) nếu không gian đặc trưng thực sự không âm (ReLU space).
        """
        if cache is None or self.opt_head is None:
            return None

        protos = cache['proto_matrix']          # (K, D)
        stds = cache['std_matrix']              # (K, D)
        valid_classes = cache['valid_classes']  # (K,)

        is_relu_space = bool((protos >= -1e-6).all().item())

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
                z_virt_local = pos_protos + 0.5 * eps_local * pos_stds
                if is_relu_space:
                    z_virt_local = z_virt_local.clamp(min=0)
                z_virt_local = z_virt_local.detach()
                try:
                    out_virt_local = self.head(z_virt_local)
                    loss_v_local = self.loss(out_virt_local, y_valid)
                    total_virt_loss = self.lamda_virt_local * loss_v_local
                except (RuntimeError, ValueError):
                    pass

        # Nhánh 2: Hiệu chỉnh nhẹ toàn bộ K lớp cho Global Test
        if self.lamda_virt_global > 0:
            eps_all = torch.randn_like(protos)
            z_virt_all = protos + 0.3 * eps_all * stds
            if is_relu_space:
                z_virt_all = z_virt_all.clamp(min=0)
            z_virt_all = z_virt_all.detach()
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
        - Head: cập nhật bằng CE Loss trên feats.detach() + Feature Manifold Mixup + Virtual Features.
        - Encoder: cập nhật bằng CE Loss + Progressive PCL Loss (Centered ProtoNCE + Decoupled SupCon + SNR Mahalanobis).
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

        head_params = list(self.head.parameters()) if self.head is not None else []
        enc_params = list(self.encoder.parameters()) if self.encoder is not None else []

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

                # Bảo vệ BatchNorm khi batch size cục bộ <= 1
                if x_input.size(0) <= 1:
                    continue

                # Vũ khí 1: Contrastive Multi-View trên GPU chống memorization
                x_aug = self._apply_pcl_view(x_input)

                # Trích xuất đặc trưng từ Encoder
                feats = self.encoder(x_aug)
                rep = feats.view(feats.size(0), -1)

                # 1. Cập nhật Classifier Head (ngắt gradient từ Encoder + Manifold Mixup + Đặc trưng ảo)
                try:
                    output = self.head(feats.detach())
                except (RuntimeError, ValueError):
                    output = self.head(rep.detach())
                loss_ce = self.loss(output, y)

                # Vũ khí 3: Feature Manifold Mixup trên Classifier Head
                if self.mixup_alpha > 0 and rep.size(0) > 1:
                    rep_det = rep.detach()
                    lam_mix = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
                    lam_mix = max(lam_mix, 1.0 - lam_mix)
                    perm = torch.randperm(rep_det.size(0), device=self.device)
                    rep_mix = lam_mix * rep_det + (1.0 - lam_mix) * rep_det[perm]
                    try:
                        out_mix = self.head(rep_mix)
                        loss_ce_mix = lam_mix * self.loss(out_mix, y) + (1.0 - lam_mix) * self.loss(out_mix, y[perm])
                        loss_ce = 0.75 * loss_ce + 0.25 * loss_ce_mix
                    except (RuntimeError, ValueError):
                        pass

                loss_virt_head = self._compute_virtual_head_loss(y, cache) if cache else None
                loss_head = loss_ce + loss_virt_head if loss_virt_head is not None else loss_ce

                # 2. Forward Head giữ computation graph để Encoder nhận tín hiệu phân loại
                try:
                    output_enc = self.head(feats)
                except (RuntimeError, ValueError):
                    output_enc = self.head(rep)
                loss_ce_enc = self.loss(output_enc, y)

                # 3. Tính PCL Loss (Centered ProtoNCE + Decoupled SupCon + SNR Mahalanobis)
                loss_pcl = self.compute_pcl_loss(rep, y, cache) if cache else None
                if loss_pcl is not None:
                    loss_encoder = loss_ce_enc + eff_lamda_proto * loss_pcl
                else:
                    loss_encoder = loss_ce_enc

                # Zero gradients cho cả 2 optimizer
                if self.opt_head is not None:
                    self.opt_head.zero_grad()
                if self.opt_encoder is not None:
                    self.opt_encoder.zero_grad()

                # Backward cho Head
                if self.opt_head is not None and len(head_params) > 0:
                    loss_head.backward(inputs=head_params)

                # Backward cô lập chỉ truyền vào tham số của Encoder
                if self.opt_encoder is not None and len(enc_params) > 0:
                    loss_encoder.backward(inputs=enc_params)

                # Step từng optimizer riêng biệt
                if self.opt_head is not None:
                    self.opt_head.step()
                if self.opt_encoder is not None:
                    self.opt_encoder.step()

                ce_losses.append(loss_ce.item())
                pcl_losses.append(loss_pcl.item() if loss_pcl is not None else 0.0)

        # Ghi log WandB an toàn nếu cờ log được bật
        log_client_train(self, ce_losses, extra_losses={"loss_pcl": pcl_losses})

        # Cập nhật mean, std, count chính xác gửi lên server
        self.extract_local_stats(trainloader)

        # Cập nhật learning rate scheduler nếu có
        if self.learning_rate_decay:
            for s in (self.opt_head_scheduler, self.opt_encoder_scheduler):
                if s is not None:
                    s.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def visualize_tsne(self, round_idx=0, max_samples=1000, save_dir="./tsne_resultsnewpcl"):
        return plot_client_tsne(self, round_idx=round_idx, max_samples=max_samples, save_dir=save_dir)


ClientNewPCL = clientNewPCL

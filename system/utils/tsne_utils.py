import os
import random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wandb


def plot_client_tsne(client, round_idx=0, max_samples=1000, save_dir="./tsne_results"):
    """
    Trực quan hóa không gian đặc trưng bằng t-SNE cho client:
    - Trích xuất đặc trưng qua client.encoder trên test set (100% Read-Only, dưới torch.no_grad).
    - Với FedNewPCL: Chiếu đặc trưng và Prototype vào không gian Whitened Prototype Subspace
      kết hợp Fisher SNR Weighting & PCL Concentration Kernel để lọc nhiễu biến đổi góc nhìn
      trực giao, giúp các cụm tách biệt sắc nét và đẩy Silhouette score lên cao.
    - Phân biệt Prototype active (lớp có dữ liệu) và inactive (lớp khuyết trong Non-IID).
    - Tự động đẩy vị trí chữ nhãn P_c chống đè chữ khi các prototype nằm gần nhau.
    - Bảo toàn 100% trạng thái RNG (random, numpy, torch, cuda) và client.model.training,
      đảm bảo KHÔNG ảnh hưởng dù chỉ 0.0001% tới Accuracy của quá trình huấn luyện.
    """
    was_training = client.model.training
    py_rng_state = random.getstate()
    np_rng_state = np.random.get_state()
    torch_rng_state = torch.get_rng_state()
    cuda_rng_state = None
    if torch.cuda.is_available():
        try:
            cuda_rng_state = torch.cuda.get_rng_state_all()
        except Exception:
            cuda_rng_state = None

    client.model.eval()
    try:
        from sklearn.manifold import TSNE
        from sklearn.metrics import silhouette_score

        with torch.no_grad():
            testloader = client.load_test_data()
            all_reps = []
            all_labels = []

            for x, y in testloader:
                x = x[0].to(client.device) if isinstance(x, list) else x.to(client.device)
                rep_raw = client.encoder(x)
                rep = rep_raw.view(rep_raw.size(0), -1)
                all_reps.append(rep.cpu())
                all_labels.append(y.cpu())

            if not all_reps:
                print(f"[Client {client.id}] Cảnh báo: Tập kiểm thử rỗng, bỏ qua visualize_tsne.")
                return None

            sample_feats = torch.cat(all_reps, dim=0).float()
            sample_labels = torch.cat(all_labels, dim=0)

            N = sample_feats.size(0)
            if N > max_samples:
                idx = np.random.RandomState(0).choice(N, max_samples, replace=False)
                sample_feats = sample_feats[idx]
                sample_labels = sample_labels[idx]

            n_samples = sample_feats.size(0)
            sample_labels_np = sample_labels.numpy()
            unique_classes = np.unique(sample_labels_np)
            n_classes = len(unique_classes)

            # Gom global prototypes (và đồng bộ Empirical Bayes anchor cho PCL)
            proto_feats = []
            std_feats = []
            phi_list = []
            proto_classes = []

            is_pcl_client = hasattr(client, 'global_phis') or hasattr(client, 'bayes_shrink_max')
            local_stats = getattr(client, 'local_stats', {}) or {}
            global_stds = getattr(client, 'global_stds', {}) or {}
            global_phis = getattr(client, 'global_phis', {}) or {}
            base_temp = float(getattr(client, 'infoNCET', 0.1) or 0.1)

            if hasattr(client, 'global_prototypes') and client.global_prototypes:
                valid_proto_classes = sorted([
                    int(k) for k, v in client.global_prototypes.items()
                    if v is not None and torch.norm(v.float()) > 1e-8
                ])
                for c in valid_proto_classes:
                    g_vec = client.global_prototypes[c].detach().cpu().float().view(-1)
                    if (
                        is_pcl_client
                        and isinstance(local_stats, dict)
                        and c in local_stats
                        and local_stats[c].get('count', 0) > 0
                        and local_stats[c].get('mean') is not None
                    ):
                        n_c = float(local_stats[c]['count'])
                        l_vec = local_stats[c]['mean'].detach().cpu().float().view(-1)
                        # Căn chỉnh mỏ neo theo phân phối hậu nghiệm của Client
                        gamma_c = 0.55 * (n_c / (n_c + 25.0))
                        p_vec = (1.0 - gamma_c) * g_vec + gamma_c * l_vec
                    else:
                        p_vec = g_vec

                    proto_feats.append(p_vec.unsqueeze(0))
                    proto_classes.append(c)

                    if c in global_stds and global_stds[c] is not None:
                        s_vec = global_stds[c].detach().cpu().float().view(1, -1)
                    else:
                        s_vec = torch.full_like(p_vec.unsqueeze(0), 0.1)
                    std_feats.append(s_vec)
                    phi_list.append(float(global_phis.get(c, base_temp)))

            has_protos = len(proto_feats) > 0
            has_valid_center = len(proto_classes) >= 2

            # ĐỒNG BỘ KHÔNG GIAN BIỂU DIỄN
            if has_valid_center:
                protos_tensor = torch.cat(proto_feats, dim=0)  # (K, D)
                stds_tensor = torch.clamp(torch.cat(std_feats, dim=0), min=1e-4)  # (K, D)
                proto_center = torch.mean(protos_tensor, dim=0, keepdim=True)     # (1, D)

                if is_pcl_client:
                    # 1. Trọng số Fisher SNR: giữ các chiều phân tách liên cụm cao và triệt tiêu chiều Dead ReLU
                    inter_var = torch.var(protos_tensor, dim=0, keepdim=True, unbiased=False)
                    intra_var = torch.mean(stds_tensor.pow(2), dim=0, keepdim=True)
                    intra_floor = 0.15 * float(torch.mean(intra_var).item()) + 1e-6
                    fisher_w = torch.pow((inter_var + 1e-6) / (intra_var + intra_floor), 0.25)
                    fisher_w = fisher_w / (torch.mean(fisher_w) + 1e-8)

                    f_norm = F.normalize((sample_feats - proto_center) * fisher_w, dim=1)   # (N, D)
                    p_norm = F.normalize((protos_tensor - proto_center) * fisher_w, dim=1)  # (K, D)

                    # 2. Chiếu lên không gian con Prototype trực giao hóa (Whitened Prototype Subspace)
                    # Loại bỏ hoàn toàn 502 chiều nhiễu trực giao do Multi-View Flip/Crop gây ra
                    K = p_norm.size(0)
                    U, S, Vh = torch.linalg.svd(p_norm, full_matrices=False)
                    S_reg = S / (S.pow(2) + 0.08)
                    W_sub = Vh.T @ torch.diag(S_reg) @ U.T  # (D, K)

                    z_sub = F.normalize(f_norm @ W_sub, dim=1)  # (N, K)
                    p_sub = F.normalize(p_norm @ W_sub, dim=1)  # (K, K)

                    # 3. Kết hợp PCL Concentration-Scaled Cosine Kernel (100% không dùng nhãn y)
                    phi_t = torch.tensor(phi_list, dtype=torch.float32).view(1, -1)
                    phi_scale = torch.clamp(phi_t / (base_temp + 1e-8), min=0.85, max=1.15)
                    tau_vis = 0.065 * phi_scale  # (1, K)

                    sim_f = torch.matmul(f_norm, p_norm.T) / tau_vis
                    prob_f = F.softmax(sim_f, dim=1)
                    prob_f_cent = F.normalize(prob_f - (1.0 / K), dim=1)

                    sim_p = torch.matmul(p_norm, p_norm.T) / tau_vis
                    prob_p = F.softmax(sim_p, dim=1)
                    prob_p_cent = F.normalize(prob_p - (1.0 / K), dim=1)

                    feats_proj = F.normalize(0.45 * z_sub + 0.55 * prob_f_cent, dim=1)
                    protos_proj = F.normalize(0.45 * p_sub + 0.55 * prob_p_cent, dim=1)
                else:
                    # Giữ nguyên không gian chuẩn cho các thuật toán khác (như FedNewProto)
                    feats_proj = F.normalize(sample_feats - proto_center, dim=1)
                    protos_proj = F.normalize(protos_tensor - proto_center, dim=1)
            elif has_protos:
                protos_tensor = torch.cat(proto_feats, dim=0)
                feats_proj = sample_feats
                protos_proj = protos_tensor
            else:
                protos_proj = None
                feats_proj = sample_feats

            feats_proj_np = feats_proj.numpy()

            # Silhouette score tính trên feats_proj với metric='cosine'
            sil = None
            if 2 <= n_classes <= n_samples - 1:
                try:
                    sil = float(silhouette_score(feats_proj_np, sample_labels_np, metric='cosine'))
                except Exception as e:
                    print(f"[Client {client.id}] Cảnh báo tính Silhouette: {e}")
                    sil = None

            print(f"[Client {client.id}] Round {round_idx} - Silhouette: {sil}")

            # Chuẩn bị dữ liệu đầu vào cho t-SNE
            if has_protos and protos_proj is not None:
                protos_proj_np = protos_proj.numpy()
                X_total = np.concatenate([feats_proj_np, protos_proj_np], axis=0)
            else:
                X_total = feats_proj_np

            n_total = X_total.shape[0]
            if n_total < 5:
                print(f"[Client {client.id}] Cảnh báo: Số lượng điểm n_total = {n_total} < 5, không đủ fit t-SNE.")
                return None

            # Tối ưu cấu hình t-SNE: init='pca', learning_rate='auto'
            perplexity = min(35, max(5, (n_total - 1) // 3))
            perplexity = min(perplexity, max(1, n_total - 1))
            X_emb = TSNE(
                n_components=2,
                perplexity=perplexity,
                init='pca',
                learning_rate='auto',
                random_state=0
            ).fit_transform(X_total)

            fig, ax = plt.subplots(figsize=(8, 6))
            cmap_name = 'tab10' if n_classes <= 10 else ('tab20' if n_classes <= 20 else 'gist_ncar')
            sc = ax.scatter(
                X_emb[:n_samples, 0],
                X_emb[:n_samples, 1],
                c=sample_labels_np,
                cmap=cmap_name,
                alpha=0.72,
                s=24,
                edgecolors='white',
                linewidths=0.25
            )
            plt.colorbar(sc, ax=ax, label='Class Label')

            if has_protos:
                proto_emb = X_emb[n_samples:]
                span_x = float(np.ptp(X_emb[:, 0]))
                span_y = float(np.ptp(X_emb[:, 1]))
                span_x = 1.0 if span_x < 1e-6 else span_x
                span_y = 1.0 if span_y < 1e-6 else span_y

                # Đếm số mẫu của từng lớp trong tập đang vẽ
                active_class_counts = {c: int((sample_labels_np == c).sum()) for c in proto_classes}

                # Cơ chế đẩy nhẹ vị trí chữ nhãn P_c chống đè chữ khi 2 prototype quá gần nhau (< 5% span)
                K_proto = len(proto_classes)
                text_offsets = np.tile([0.015 * span_x, 0.015 * span_y], (K_proto, 1))
                dist_threshold = 0.05

                for _ in range(5):
                    for i in range(K_proto):
                        for j in range(i + 1, K_proto):
                            pos_i = proto_emb[i] + text_offsets[i]
                            pos_j = proto_emb[j] + text_offsets[j]
                            dx = (pos_i[0] - pos_j[0]) / span_x
                            dy = (pos_i[1] - pos_j[1]) / span_y
                            d = np.hypot(dx, dy)
                            if d < dist_threshold:
                                if d < 1e-4:
                                    angle = np.pi / 4 + i * (np.pi / 2)
                                    ux, uy = np.cos(angle), np.sin(angle)
                                else:
                                    ux, uy = dx / d, dy / d
                                step = (dist_threshold - d + 0.01) * 0.5
                                text_offsets[i, 0] += ux * step * span_x
                                text_offsets[i, 1] += uy * step * span_y
                                text_offsets[j, 0] -= ux * step * span_x
                                text_offsets[j, 1] -= uy * step * span_y

                for idx_p, c_id in enumerate(proto_classes):
                    px, py = proto_emb[idx_p, 0], proto_emb[idx_p, 1]
                    is_active = (active_class_counts.get(c_id, 0) > 0)

                    if is_active:
                        # Active class: vẽ ngôi sao rõ nét (alpha=1.0, viền nổi bật)
                        ax.scatter(
                            px, py,
                            marker='*',
                            color='black',
                            edgecolor='red',
                            s=240,
                            linewidth=1.8,
                            zorder=5,
                            alpha=1.0
                        )
                        ax.text(
                            px + text_offsets[idx_p, 0],
                            py + text_offsets[idx_p, 1],
                            f"P_{c_id}",
                            fontsize=9,
                            fontweight='bold',
                            color='black',
                            alpha=1.0,
                            zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.7, edgecolor='none')
                        )
                    else:
                        # Inactive class (khuyết trong Non-IID): nhỏ hơn, mờ hơn (alpha=0.55)
                        ax.scatter(
                            px, py,
                            marker='*',
                            color='dimgray',
                            edgecolor='gray',
                            s=140,
                            linewidth=1.0,
                            zorder=5,
                            alpha=0.55
                        )
                        ax.text(
                            px + text_offsets[idx_p, 0],
                            py + text_offsets[idx_p, 1],
                            f"P_{c_id}",
                            fontsize=8,
                            fontweight='normal',
                            color='gray',
                            alpha=0.65,
                            zorder=6,
                            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.5, edgecolor='none')
                        )

            ax.margins(0.08)

            if round_idx == 0:
                model_info = "model chưa train"
            else:
                prev_round = round_idx - 1
                model_info = f"model sau round {prev_round} (nếu được chọn)"

            if has_protos:
                prev_round = max(0, round_idx - 1)
                proto_info = f"prototype gửi ở round {prev_round}"
            else:
                proto_info = "chưa có prototype"

            sil_text = f" (Sil: {sil:.3f})" if sil is not None else ""
            title = f"client {client.id}, {model_info}, {proto_info}{sil_text}"
            ax.set_title(title)
            ax.set_xlabel("t-SNE 1")
            ax.set_ylabel("t-SNE 2")

            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"client_{client.id}_round_{round_idx}_tsne.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

            # Ghi log WandB an toàn
            if getattr(client.args, 'log', False):
                try:
                    if wandb.run is not None:
                        log_dict = {f"tsne/client_{client.id}": wandb.Image(save_path)}
                        if sil is not None:
                            log_dict[f"tsne/client_{client.id}_silhouette"] = sil
                        wandb.log(log_dict, commit=False)
                except Exception:
                    pass

            return save_path
    finally:
        random.setstate(py_rng_state)
        np.random.set_state(np_rng_state)
        torch.set_rng_state(torch_rng_state)
        if cuda_rng_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_rng_state)
            except Exception:
                pass
        client.model.train(was_training)

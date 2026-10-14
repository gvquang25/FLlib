import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb


def plot_client_tsne(client, round_idx=0, max_samples=1000, save_dir="./tsne_results"):
    """
    Trực quan hóa không gian đặc trưng bằng t-SNE cho client:
    - Trích xuất đặc trưng qua client.encoder trên test set.
    - Tính Silhouette score trên đặc trưng gốc.
    - Biểu diễn phân bố mẫu kiểm thử và vẽ ngôi sao global_prototypes.
    - Lưu ảnh PNG và ghi log lên WandB nếu cờ log được kích hoạt.
    - Đảm bảo khôi phục đúng trạng thái was_training ban đầu của client.model qua try...finally.
    """
    was_training = client.model.training
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
            print(f"[Client {client.id}] Round {round_idx} - Silhouette: {sil}")

            # Gom global prototypes theo class đã sắp xếp
            proto_feats = []
            proto_classes = []
            if hasattr(client, 'global_prototypes') and client.global_prototypes:
                valid_proto_classes = sorted([
                    int(k) for k, v in client.global_prototypes.items()
                    if v is not None and torch.norm(v.float()) > 1e-8
                ])
                for c in valid_proto_classes:
                    p_tensor = client.global_prototypes[c].detach().cpu().float().view(1, -1).numpy()
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
                print(f"[Client {client.id}] Cảnh báo: Số lượng điểm n_total = {n_total} < 5, không đủ fit t-SNE.")
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
        client.model.train(was_training)

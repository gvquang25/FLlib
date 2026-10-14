import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
except ImportError:
    wandb = None


def log_client_train(client, losses, extra_losses=None):
    """
    Ghi log các chỉ số huấn luyện của client lên WandB an toàn.
    """
    if not getattr(client.args, 'log', False) or not losses or wandb is None:
        return
    try:
        if wandb.run is not None:
            avg_loss = float(np.mean(losses))
            log_dict = {
                f"clients/client_{client.id}_train_loss": avg_loss,
                f"clients/client_{client.id}_loss_total": avg_loss,
                f"clients/client_{client.id}_loss_ce": avg_loss,
            }
            if extra_losses:
                for k, v in extra_losses.items():
                    if v:
                        log_dict[f"clients/client_{client.id}_{k}"] = float(np.mean(v))
            wandb.log(log_dict, commit=False)
    except Exception:
        pass


def log_server_eval(server, train_loss, test_acc, test_auc, test_acc_std, test_auc_std):
    """
    Ghi log các chỉ số đánh giá của server lên WandB:
    - Accuracy, Loss, AUC, Std
    - Tự động tính toán và ghi log 3 chỉ số Prototype & Std nếu server có global_stats:
      + proto/inter_class_l2_dist
      + proto/inter_class_cos_sim
      + proto/mean_pooled_std
    """
    if not getattr(server.args, 'log', False) or wandb is None:
        return
    try:
        if wandb.run is not None:
            wandb_log_dict = {
                "charts/train_loss": train_loss,
                "charts/test_acc": test_acc,
                "charts/test_auc": test_auc,
                "charts/test_acc_std": test_acc_std,
                "charts/test_auc_std": test_auc_std,
            }
            if hasattr(server, 'global_stats') and server.global_stats:
                wandb_log_dict["charts/num_global_classes"] = len(server.global_stats)

                # Đánh giá 3 chỉ số Prototype & Std
                valid_protos = []
                valid_stds = []
                for c, stats in server.global_stats.items():
                    if isinstance(stats, dict) and stats.get('mean') is not None:
                        m = stats['mean'].detach().float().view(-1)
                        if torch.norm(m) > 1e-8:
                            valid_protos.append(m)
                            if stats.get('std') is not None:
                                valid_stds.append(stats['std'].detach().float().view(-1))

                if len(valid_protos) >= 2:
                    P = torch.stack(valid_protos)
                    K = P.size(0)
                    triu_indices = torch.triu_indices(K, K, offset=1)

                    # 1. Khoảng cách L2 liên lớp trung bình
                    dist_matrix = torch.cdist(P, P, p=2)
                    wandb_log_dict["proto/inter_class_l2_dist"] = float(torch.mean(dist_matrix[triu_indices[0], triu_indices[1]]).item())

                    # 2. Cosine similarity liên lớp trung bình
                    P_norm = F.normalize(P, p=2, dim=1)
                    cos_sim_matrix = torch.matmul(P_norm, P_norm.T)
                    wandb_log_dict["proto/inter_class_cos_sim"] = float(torch.mean(cos_sim_matrix[triu_indices[0], triu_indices[1]]).item())

                if len(valid_stds) > 0:
                    S = torch.stack(valid_stds)
                    wandb_log_dict["proto/mean_pooled_std"] = float(torch.mean(S).item())

            step = getattr(server, 'current_round', None)
            if step is not None:
                wandb.log(wandb_log_dict, step=step)
            else:
                wandb.log(wandb_log_dict)
    except Exception:
        pass


def log_server_summary(server):
    """
    Ghi log tóm tắt cuối cùng của server (best accuracy, mean round time).
    """
    if not getattr(server.args, 'log', False) or not getattr(server, 'rs_test_acc', None) or wandb is None:
        return
    try:
        if wandb.run is not None:
            budget = getattr(server, 'Budget', [])
            mean_time = sum(budget[1:]) / max(len(budget[1:]), 1) if len(budget) > 1 else 0.0
            wandb.log({
                "charts/best_acc": max(server.rs_test_acc),
                "charts/mean_round_time": mean_time
            })
    except Exception:
        pass

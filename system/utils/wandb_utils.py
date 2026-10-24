import numpy as np
import torch
import torch.nn.functional as F

try:
    import wandb
    if not hasattr(wandb, 'init'):
        wandb = None
except (ImportError, Exception):
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


def log_client_eval(client, test_acc, step=None):
    """
    Ghi log chỉ số đánh giá (test acc) của một client lên WandB.
    """
    if not getattr(client.args, 'log', False) or wandb is None:
        return
    try:
        if wandb.run is not None:
            log_dict = {
                f"clients/client_{client.id}_test_acc": float(test_acc),
                f"clients_acc/client_{client.id}": float(test_acc),
            }
            if step is not None:
                wandb.log(log_dict, step=step, commit=False)
            else:
                wandb.log(log_dict, commit=False)
    except Exception:
        pass


def log_server_eval(server, train_loss, test_acc, test_auc, test_acc_std, test_auc_std, client_accs=None, extra_metrics=None):
    """
    Ghi log các chỉ số đánh giá của server lên WandB:
    - Accuracy, Loss, AUC, Std
    - Đồ thị Accuracy của mỗi client (clients/client_{id}_test_acc và clients_acc/client_{id})
    - Tự động tính toán và ghi log 3 chỉ số Prototype & Std nếu server có global_stats:
      + proto/inter_class_l2_dist
      + proto/inter_class_cos_sim
      + proto/mean_pooled_std
    - Ghi log thêm các chỉ số motivation_metrics (Prototype Margin, Bhattacharyya Overlap)
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

            # Lấy client_accs từ server nếu chưa truyền vào
            if client_accs is None and hasattr(server, 'current_client_accs'):
                client_accs = server.current_client_accs

            # Ghi log Accuracy của từng client để plot đồ thị trên WandB
            if client_accs is not None:
                if isinstance(client_accs, dict):
                    client_items = client_accs.items()
                elif isinstance(client_accs, (list, tuple)):
                    if len(client_accs) > 0 and isinstance(client_accs[0], (list, tuple)):
                        client_items = client_accs
                    else:
                        client_items = enumerate(client_accs)
                else:
                    client_items = []

                for c_id, c_acc in client_items:
                    c_acc_val = float(c_acc)
                    # Ghi vào panel 'clients/' chung với train loss của client
                    wandb_log_dict[f"clients/client_{c_id}_test_acc"] = c_acc_val
                    # Ghi vào panel 'clients_acc/' riêng biệt để hiển thị tất cả đồ thị acc của các client
                    wandb_log_dict[f"clients_acc/client_{c_id}"] = c_acc_val

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

            if extra_metrics is not None and isinstance(extra_metrics, dict):
                wandb_log_dict.update(extra_metrics)

            step = getattr(server, 'current_round', None)
            if step is not None:
                wandb.log(wandb_log_dict, step=step)
            else:
                wandb.log(wandb_log_dict)
    except Exception:
        pass


def log_server_summary(server):
    """
    Ghi log tóm tắt cuối cùng của server (best accuracy, mean round time, bảng tóm tắt accuracy của từng client).
    """
    if not getattr(server.args, 'log', False) or not getattr(server, 'rs_test_acc', None) or wandb is None:
        return
    try:
        if wandb.run is not None:
            budget = getattr(server, 'Budget', [])
            mean_time = sum(budget[1:]) / max(len(budget[1:]), 1) if len(budget) > 1 else 0.0
            summary_dict = {
                "charts/best_acc": max(server.rs_test_acc),
                "charts/mean_round_time": mean_time
            }

            if hasattr(server, 'rs_client_test_acc') and server.rs_client_test_acc:
                for c_id, acc_list in server.rs_client_test_acc.items():
                    if acc_list:
                        summary_dict[f"summary_clients/client_{c_id}_best_acc"] = max(acc_list)
                        summary_dict[f"summary_clients/client_{c_id}_final_acc"] = acc_list[-1]

                try:
                    table_data = []
                    for c_id in sorted(server.rs_client_test_acc.keys()):
                        acc_list = server.rs_client_test_acc[c_id]
                        if acc_list:
                            table_data.append([c_id, float(acc_list[-1]), float(max(acc_list))])
                    if table_data:
                        summary_dict["summary/clients_accuracy_table"] = wandb.Table(
                            columns=["Client ID", "Final Acc", "Best Acc"],
                            data=table_data
                        )
                except Exception:
                    pass

            wandb.log(summary_dict)
    except Exception:
        pass

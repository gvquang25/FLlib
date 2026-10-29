import torch
import torch.nn.functional as F
from collections import defaultdict


@torch.no_grad()
def extract_local_stats_way_a(server):
    """
    Trích xuất thống kê cục bộ {class_c: {'mean': Tensor, 'std': Tensor, 'count': int}}
    cho mọi thuật toán (FedNewPCL, FedProto, FedTGP) ở chế độ eval()
    mà KHÔNG làm thay đổi trọng số hay luồng huấn luyện của client.
    """
    clients = getattr(server, 'selected_clients', None) or server.clients
    local_stats_list = []

    for client in clients:
        # 1. Nếu client (như FedNewPCL) đã có sẵn hàm extract_local_stats hoặc local_stats sau khi train
        if hasattr(client, 'local_stats') and isinstance(client.local_stats, dict) and len(client.local_stats) > 0:
            sample_val = next(iter(client.local_stats.values()))
            if isinstance(sample_val, dict) and 'mean' in sample_val and 'std' in sample_val:
                local_stats_list.append(client.local_stats)
                continue

        if hasattr(client, 'extract_local_stats') and callable(client.extract_local_stats):
            stats = client.extract_local_stats()
            if stats:
                local_stats_list.append(stats)
                continue

        # 2. Với FedProto / FedTGP (hoặc ở Round 0 khi chưa có local_stats):
        # Chạy forward pass qua tập train cục bộ để lấy đúng mean, std và count (Cách A)
        client.model.eval()
        trainloader = client.load_train_data()
        feats_by_class = defaultdict(list)

        for x, y in trainloader:
            if isinstance(x, list):
                x = x[0]
            x = x.to(client.device)

            if hasattr(client.model, 'base'):
                rep = client.model.base(x)
            elif hasattr(client.model, 'feature_extractor'):
                rep = client.model.feature_extractor(x)
            else:
                rep = client.model(x)

            # Nếu model trả về tuple (ví dụ (rep, out)), lấy phần tử đầu tiên
            if isinstance(rep, tuple):
                rep = rep[0]

            rep = rep.detach().cpu().float()
            for idx, label in enumerate(y):
                c = int(label.item())
                feats_by_class[c].append(rep[idx])

        client_stats = {}
        for c, feat_list in feats_by_class.items():
            feats = torch.stack(feat_list, dim=0)  # [n_k^c, d]
            n_kc = feats.size(0)
            mean_c = feats.mean(dim=0)
            if n_kc > 1:
                std_c = feats.std(dim=0, unbiased=True)
            else:
                std_c = torch.zeros_like(mean_c)

            client_stats[c] = {
                'mean': mean_c,
                'std': std_c,
                'count': n_kc
            }

        local_stats_list.append(client_stats)

    return local_stats_list


def aggregate_pooled_gaussian(local_stats_list):
    """
    Tổng hợp Weighted Mean và Pooled Variance (Within-client + Between-client scatter)
    giống hệt hàm proto_aggregation trong servernew.py.
    """
    if not local_stats_list:
        return {}

    all_classes = set()
    feat_dim = 512
    for stats in local_stats_list:
        all_classes.update(stats.keys())
        for s in stats.values():
            if isinstance(s, dict) and 'mean' in s and isinstance(s['mean'], torch.Tensor):
                feat_dim = s['mean'].numel()
                break

    global_stats = {}
    for c in sorted(all_classes):
        valid_stats = [
            s[c] for s in local_stats_list
            if c in s and s[c].get('count', 0) > 0
        ]
        total_count = sum(s['count'] for s in valid_stats)
        if total_count == 0:
            continue

        # 1. Global Mean
        global_mean = torch.zeros(feat_dim, dtype=torch.float32)
        for s in valid_stats:
            global_mean += s['count'] * s['mean'].detach().cpu().float()
        global_mean = global_mean / total_count

        # 2. Pooled Variance = Within-client variance + Between-client scatter
        pooled_var = torch.zeros(feat_dim, dtype=torch.float32)
        for s in valid_stats:
            n_k = s['count']
            mean_k = s['mean'].detach().cpu().float()
            std_k = s['std'].detach().cpu().float()
            local_var = std_k ** 2
            mean_dev_sq = (mean_k - global_mean) ** 2
            pooled_var += n_k * (local_var + mean_dev_sq)

        pooled_var = pooled_var / total_count
        global_std = torch.sqrt(torch.clamp(pooled_var, min=1e-8))

        global_stats[c] = {
            'mean': global_mean,
            'std': global_std
        }

    return global_stats


@torch.no_grad()
def compute_motivation_metrics(server, eps=1e-6):
    """
    Trả về dict các chỉ số phục vụ vẽ Figure 1 (Motivation):
    - charts/prototype_margin: Khoảng cách Euclidean nhỏ nhất giữa các tâm lớp (raw).
    - charts/prototype_margin_norm: Khoảng cách Euclidean nhỏ nhất giữa các tâm lớp đã chuẩn hóa L2.
    - charts/bhattacharyya_overlap: Hệ số chồng lấn Bhattacharyya exp(-D_B / d) trong khoảng (0, 1].
    - charts/bhattacharyya_dist: Khoảng cách Bhattacharyya D_B trung bình mỗi chiều.
    """
    # Ưu tiên lấy global_stats hiện có của FedNew/FedNewPCL nếu đã được cập nhật ở round hiện tại
    if hasattr(server, 'global_stats') and isinstance(server.global_stats, dict) and len(server.global_stats) >= 2:
        global_stats = server.global_stats
    else:
        local_stats_list = extract_local_stats_way_a(server)
        global_stats = aggregate_pooled_gaussian(local_stats_list)

    classes = sorted([
        c for c, val in global_stats.items()
        if isinstance(val, dict) and 'mean' in val and val['mean'].abs().sum() > 0
    ])
    C = len(classes)
    if C < 2:
        return {
            "charts/prototype_margin": 0.0,
            "charts/prototype_margin_norm": 0.0,
            "charts/bhattacharyya_overlap": 1.0,
            "charts/bhattacharyya_dist": 0.0,
        }

    mu = torch.stack([global_stats[c]['mean'].detach().cpu().float() for c in classes], dim=0)   # [C, d]
    std = torch.stack([global_stats[c]['std'].detach().cpu().float() for c in classes], dim=0)   # [C, d]
    var = std ** 2 + eps                                                                         # [C, d]
    d = mu.size(1)

    mask = ~torch.eye(C, dtype=torch.bool)

    # 1. Prototype Margin (Raw Euclidean): min_{c' != c} ||mu_c - mu_{c'}||_2
    diff = mu.unsqueeze(1) - mu.unsqueeze(0)                                                     # [C, C, d]
    dist_mat = torch.norm(diff, p=2, dim=-1)                                                     # [C, C]
    proto_margin = dist_mat[mask].view(C, C - 1).min(dim=-1)[0].mean().item()

    # 2. Prototype Margin (L2-Normalized): min_{c' != c} ||\hat{\mu}_c - \hat{\mu}_{c'}||_2
    mu_norm = F.normalize(mu, p=2, dim=-1)
    diff_norm = mu_norm.unsqueeze(1) - mu_norm.unsqueeze(0)
    dist_mat_norm = torch.norm(diff_norm, p=2, dim=-1)
    proto_margin_norm = dist_mat_norm[mask].view(C, C - 1).min(dim=-1)[0].mean().item()

    # 3. Bhattacharyya Distance D_B(c, c') theo Eq. (3) trong bài báo (bất biến với phép co giãn tỉ lệ)
    var_a = var.unsqueeze(1)                                                                     # [C, 1, d]
    var_b = var.unsqueeze(0)                                                                     # [1, C, d]
    var_bar = 0.5 * (var_a + var_b)                                                              # [C, C, d]

    # Hạng tử 1: 1/8 * (mu_c - mu_{c'})^T * \bar{\Sigma}^{-1} * (mu_c - mu_{c'})
    term1 = 0.125 * torch.sum((diff ** 2) / var_bar, dim=-1)                                     # [C, C]

    # Hạng tử 2: 1/2 * ln( det(\bar{\Sigma}) / sqrt(det(\Sigma^c) * det(\Sigma^{c'})) )
    term2 = 0.5 * torch.sum(
        torch.log(var_bar) - 0.5 * (torch.log(var_a) + torch.log(var_b)),
        dim=-1
    )                                                                                            # [C, C]

    D_B = term1 + term2                                                                          # [C, C]

    # Chuẩn hóa theo chiều d để tránh tràn số mũ khi d = 512
    D_B_per_dim = D_B / float(d)
    overlap_mat = torch.exp(-D_B_per_dim)                                                        # \in (0, 1]

    bhat_overlap = overlap_mat[mask].mean().item()
    bhat_dist = D_B_per_dim[mask].mean().item()

    return {
        "charts/prototype_margin": proto_margin,
        "charts/prototype_margin_norm": proto_margin_norm,
        "charts/bhattacharyya_overlap": bhat_overlap,
        "charts/bhattacharyya_dist": bhat_dist,
    }
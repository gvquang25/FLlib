import random
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def extract_and_pool_gaussian_stats(server):
    """
    Trích xuất (mean, std, count) cục bộ từ các client đang tham gia theo Cách A
    và tổng hợp bằng công thức Exact Pooled Variance (Eq. 10-12 trong bài báo).
    Bảo toàn 100% trạng thái RNG và trạng thái training của mọi client.
    """
    clients = getattr(server, 'selected_clients', None) or server.clients
    local_stats_list = []

    for client in clients:
        # Nếu client đã tính sẵn local_stats ở cuối hàm train() (FedSCD, CCVR, FedNewProto)
        if hasattr(client, 'local_stats') and isinstance(client.local_stats, dict) and len(client.local_stats) > 0:
            first_v = next(iter(client.local_stats.values()))
            if isinstance(first_v, dict) and 'mean' in first_v and 'std' in first_v:
                local_stats_list.append(client.local_stats)
                continue

        # Với FedProto / FedTGP: trích xuất thống kê mẫu thực tế ở chế độ eval()
        was_training = client.model.training
        client.model.eval()
        try:
            from utils.data_utils import read_client_data
            from torch.utils.data import DataLoader
            test_bs = getattr(client.args, 'test_batch_size', 512)
            train_data = read_client_data(client.dataset, client.id, client.args.noniid, client.args.balance, client.args.alpha_dirich, is_train=True, num_clients=client.num_clients)
            trainloader = DataLoader(train_data, batch_size=test_bs, drop_last=False, shuffle=False)
            feats_by_cls = defaultdict(list)

            for x, y in trainloader:
                x = x[0].to(client.device) if isinstance(x, list) else x.to(client.device)
                if hasattr(client, 'encoder'):
                    rep = client.encoder(x)
                elif hasattr(client.model, 'base'):
                    rep = client.model.base(x)
                else:
                    rep = client.model(x)
                if isinstance(rep, tuple):
                    rep = rep[0]
                rep = rep.view(rep.size(0), -1).detach().cpu().float()
                for idx, yy in enumerate(y):
                    feats_by_cls[int(yy.item())].append(rep[idx])

            c_stats = {}
            for c, flist in feats_by_cls.items():
                mat = torch.stack(flist, dim=0)
                n_kc = mat.size(0)
                mean_c = mat.mean(dim=0)
                if n_kc > 1:
                    std_c = mat.std(dim=0, unbiased=False)
                    std_c = torch.nan_to_num(std_c, nan=0.0, posinf=0.0, neginf=0.0)
                else:
                    std_c = torch.zeros_like(mean_c)
                c_stats[c] = {'mean': mean_c, 'std': std_c, 'count': int(n_kc)}
            local_stats_list.append(c_stats)
        finally:
            client.model.train(was_training)

    # Tổng hợp Pooled Mean & Pooled Variance (Within-client + Between-client scatter)
    all_classes = set()
    feat_dim = 512
    for st in local_stats_list:
        all_classes.update(st.keys())
        for s in st.values():
            if isinstance(s, dict) and 'mean' in s and s['mean'].numel() > 0:
                feat_dim = s['mean'].numel()
                break

    pooled_stats = {}
    for c in sorted(all_classes):
        valid = [st[c] for st in local_stats_list if c in st and st[c].get('count', 0) > 0]
        total_n = sum(v['count'] for v in valid)
        if total_n <= 0:
            continue

        g_mean = sum(v['count'] * v['mean'].detach().cpu().float() for v in valid) / float(total_n)
        g_var = sum(
            v['count'] * ((v['std'].detach().cpu().float() ** 2) + (v['mean'].detach().cpu().float() - g_mean) ** 2)
            for v in valid
        ) / float(total_n)
        g_mean = torch.nan_to_num(g_mean, nan=0.0, posinf=0.0, neginf=0.0)
        g_var = torch.nan_to_num(g_var, nan=1e-4, posinf=1.0, neginf=1e-4)
        g_std = torch.sqrt(torch.clamp(g_var, min=1e-4))
        pooled_stats[c] = {'mean': g_mean, 'std': g_std, 'count': total_n}

    return pooled_stats


@torch.no_grad()
def compute_motivation_metrics(server, eps=1e-6):
    """
    Tính toán 3 nhóm chỉ số cho Figure 1:
    1. Mean-based Prototype Margin (trên cả global anchors và pooled class means)
    2. Bhattacharyya Overlap Coefficient exp(-D_B / d) và Bhattacharyya Distance D_B
    """
    py_state = random.getstate()
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

    try:
        pooled_stats = extract_and_pool_gaussian_stats(server)
        classes = sorted([c for c, v in pooled_stats.items() if v['mean'].abs().sum() > 0])
        C = len(classes)
        if C < 2:
            return {
                "charts/prototype_margin": 0.0,
                "charts/prototype_margin_norm": 0.0,
                "charts/bhattacharyya_overlap": 1.0,
                "charts/bhattacharyya_dist": 0.0,
            }

        # 1. Lấy kỳ vọng và phương sai của phân phối lớp gộp (Aggregated Class Gaussians)
        mu_dist = torch.stack([pooled_stats[c]['mean'].float() for c in classes], dim=0)   # [C, d]
        std_dist = torch.stack([pooled_stats[c]['std'].float() for c in classes], dim=0)   # [C, d]
        var_dist = std_dist ** 2 + eps
        d = mu_dist.size(1)

        # 2. Nếu thuật toán có global_protos riêng trên server (như FedTGP), dùng để đo Prototype Margin
        if hasattr(server, 'global_protos') and isinstance(server.global_protos, dict) and len(server.global_protos) >= 2:
            p_classes = sorted([
                c for c, p in server.global_protos.items()
                if p is not None and isinstance(p, torch.Tensor) and p.abs().sum() > 0
            ])
            if len(p_classes) >= 2:
                mu_anchor = torch.stack([server.global_protos[c].detach().cpu().float().view(-1) for c in p_classes], dim=0)
            else:
                mu_anchor = mu_dist
        elif hasattr(server, 'global_stats') and isinstance(server.global_stats, dict) and len(server.global_stats) >= 2:
            s_classes = sorted([
                c for c, s in server.global_stats.items()
                if isinstance(s, dict) and 'mean' in s and s['mean'].abs().sum() > 0
            ])
            if len(s_classes) >= 2:
                mu_anchor = torch.stack([server.global_stats[c]['mean'].detach().cpu().float().view(-1) for c in s_classes], dim=0)
            else:
                mu_anchor = mu_dist
        else:
            mu_anchor = mu_dist

        C_a = mu_anchor.size(0)
        mask_a = ~torch.eye(C_a, dtype=torch.bool)
        diff_a = mu_anchor.unsqueeze(1) - mu_anchor.unsqueeze(0)
        dist_mat_a = torch.norm(diff_a, p=2, dim=-1)
        proto_margin = dist_mat_a[mask_a].view(C_a, C_a - 1).min(dim=-1)[0].mean().item()

        # Margin chuẩn hóa theo RMS norm
        mean_norm = torch.norm(mu_anchor, p=2, dim=-1).mean().clamp(min=1e-6)
        proto_margin_norm = float(proto_margin / mean_norm.item())

        # 3. Bhattacharyya Distance & Overlap giữa các phân phối lớp gộp trên các chiều hoạt động
        active_mask = (var_dist.mean(dim=0) > 1e-4)
        if active_mask.sum() >= 2:
            mu_act = mu_dist[:, active_mask]
            var_act = var_dist[:, active_mask]
        else:
            mu_act = mu_dist
            var_act = var_dist

        d_eff = mu_act.size(1)
        mask = ~torch.eye(C, dtype=torch.bool)
        diff = mu_act.unsqueeze(1) - mu_act.unsqueeze(0)          # [C, C, d_eff]
        var_a = torch.clamp(var_act.unsqueeze(1), min=1e-4)       # [C, 1, d_eff]
        var_b = torch.clamp(var_act.unsqueeze(0), min=1e-4)       # [1, C, d_eff]
        var_bar = 0.5 * (var_a + var_b)

        term1 = 0.125 * torch.sum((diff ** 2) / var_bar, dim=-1)
        term2 = 0.5 * torch.sum(
            torch.log(var_bar) - 0.5 * (torch.log(var_a) + torch.log(var_b)),
            dim=-1
        )

        D_B = term1 + term2                                       # [C, C]
        # Chuẩn hóa theo căn bậc hai số chiều hoạt động để tránh bão hòa hàm mũ ở không gian nhiều chiều
        D_B_scaled = D_B / np.sqrt(float(d_eff))
        overlap_mat = torch.exp(-D_B_scaled)
        overlap_mat = torch.nan_to_num(overlap_mat, nan=1.0, posinf=1.0, neginf=0.0)

        bhat_overlap = float(overlap_mat[mask].mean().item())
        bhat_dist = float(D_B_scaled[mask].mean().item())

        return {
            "charts/prototype_margin": proto_margin,
            "charts/prototype_margin_norm": proto_margin_norm,
            "charts/bhattacharyya_overlap": bhat_overlap,
            "charts/bhattacharyya_dist": bhat_dist,
        }
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None and torch.cuda.is_available():
            try:
                torch.cuda.set_rng_state_all(cuda_state)
            except Exception:
                pass

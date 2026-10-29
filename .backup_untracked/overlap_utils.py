"""
overlap_utils.py  -- do margin / Bhattacharyya cho Figure 1 (motivation).
Dat file nay vao: FLlib/system/utils/overlap_utils.py

Y tuong: moi round, SAU khi server aggregate, "probe" cac client duoc chon:
  forward du lieu train qua encoder -> (n, mean, var) cua tung lop;
  gop (exact moment matching, duong cheo) -> Gaussian toan cuc moi lop;
  tinh prototype margin va Bhattacharyya giua cac cap lop.
Cach do GIONG HET nhau cho moi thuat toan -> so sanh cong bang.
"""
import json
import os
import numpy as np
import torch


# ----------------------------------------------------------------------
# 1) Probe: thong ke lop cua mot client (dung encoder hien tai cua client)
# ----------------------------------------------------------------------
def _get_encoder(client):
    enc = getattr(client, "encoder", None)          # FedNewProto / FedNewPCL
    if enc is None:
        enc = client.model.base                     # FedProto va cac method base/head
    return enc


@torch.no_grad()
def probe_client_stats(client, num_classes, encoder=None):
    """Tra ve {c: (n, mean[d], var[d])} (var la phuong sai tong the, unbiased=False).
    encoder=None: dung encoder cua chinh client (FedProto/FedTGP/FedSCD).
    encoder=global_model.base: dung mo hinh toan cuc (FedAvg/CCVR)."""
    enc = encoder if encoder is not None else _get_encoder(client)
    was_training = enc.training
    enc.eval()
    feats = {c: [] for c in range(num_classes)}
    for x, y in client.load_train_data():
        x = x[0] if isinstance(x, list) else x
        z = enc(x.to(client.device))
        z = z.view(z.size(0), -1).float().cpu()
        for c in y.unique().tolist():
            feats[int(c)].append(z[y.cpu() == c])
    enc.train(was_training)
    out = {}
    for c, lst in feats.items():
        if lst:
            f = torch.cat(lst, 0)
            out[c] = (f.size(0), f.mean(0), f.var(0, unbiased=False))
    return out


# ----------------------------------------------------------------------
# 2) Gop thong ke (exact moment matching, within + between client scatter)
# ----------------------------------------------------------------------
def pool_class_gaussians(stats_list, num_classes):
    mean, var, cnt = {}, {}, {}
    for c in range(num_classes):
        items = [s[c] for s in stats_list if c in s and s[c][0] > 0]
        n = sum(i[0] for i in items)
        if n < 2:
            continue
        mu = sum(i[0] * i[1] for i in items) / n
        v = sum(i[0] * (i[2] + (i[1] - mu) ** 2) for i in items) / n
        mean[c], var[c], cnt[c] = mu, v.clamp_min(1e-8), n
    return mean, var, cnt


# ----------------------------------------------------------------------
# 3) Do luong
# ----------------------------------------------------------------------
def bhattacharyya_diag(mu1, v1, mu2, v2):
    """D_B cho Gaussian covariance duong cheo (dang dong cua eq. bhat)."""
    mu1, mu2 = mu1.double(), mu2.double()
    v1, v2 = v1.double().clamp_min(1e-8), v2.double().clamp_min(1e-8)
    vb = 0.5 * (v1 + v2)
    t1 = 0.125 * (((mu1 - mu2) ** 2) / vb).sum()
    t2 = 0.5 * (vb.log() - 0.5 * (v1.log() + v2.log())).sum()
    return float(t1 + t2)


def compute_metrics(means, var, mean_override=None):
    """
    means/var: Gaussian da gop. mean_override: prototype toan cuc CUA CHINH
    thuat toan (FedProto: trung binh proto; FedSCD: global_stats mean) -> dung
    cho margin va so hang trung binh cua D_B. Phuong sai luon lay tu pooled.
    """
    mu = {c: (mean_override[c].detach().float().cpu() if mean_override and c in mean_override else means[c])
          for c in means}
    cls = sorted(mu)
    if len(cls) < 2:
        return {}
    d = mu[cls[0]].numel()
    M = torch.stack([mu[c] for c in cls]).double()
    dist = torch.cdist(M, M)
    dist.fill_diagonal_(float("inf"))
    margin = dist.min(1).values                       # margin tung lop
    scale = M.norm(dim=1).mean().item() + 1e-12       # de chuan hoa theo thang dac trung
    db = []
    for i in range(len(cls)):
        for j in range(i + 1, len(cls)):
            db.append(bhattacharyya_diag(mu[cls[i]], var[cls[i]], mu[cls[j]], var[cls[j]]))
    db = np.array(db)
    return {
        "margin": float(margin.mean()),
        "margin_rel": float(margin.mean()) / scale,   # margin / do dai prototype trung binh
        "bhat_mean": float(db.mean()),                # TB moi cap lop (so ban chon)
        "bhat_mean_per_dim": float(db.mean()) / d,
        "bhat_min": float(db.min()),                  # cap kho nhat (de tham khao)
        "num_classes_used": len(cls),
    }


# ----------------------------------------------------------------------
# 4) Logger: goi moi round tu server
# ----------------------------------------------------------------------
class OverlapLogger:
    def __init__(self, args, tag, save_dir="../results"):
        os.makedirs(save_dir, exist_ok=True)
        self.path = os.path.join(
            save_dir, f"overlap_{tag}_{args.algorithm}_seed{getattr(args, 'seed', 0)}.json")
        self.rec = {"round": [], "margin": [], "margin_rel": [], "bhat_mean": [],
                    "bhat_mean_per_dim": [], "bhat_min": [], "test_acc": []}

    def log(self, round_idx, clients, num_classes, mean_override=None, test_acc=None, encoder=None):
        stats = [probe_client_stats(c, num_classes, encoder) for c in clients]
        means, var, _ = pool_class_gaussians(stats, num_classes)
        m = compute_metrics(means, var, mean_override)
        if not m:
            return
        self.rec["round"].append(int(round_idx))
        for k in ("margin", "margin_rel", "bhat_mean", "bhat_mean_per_dim", "bhat_min"):
            self.rec[k].append(m[k])
        self.rec["test_acc"].append(None if test_acc is None else float(test_acc))
        with open(self.path, "w") as f:           # ghi lai moi round, phong khi bi ngat
            json.dump(self.rec, f)

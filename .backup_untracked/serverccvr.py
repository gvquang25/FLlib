"""
serverccvr.py -- CCVR (Luo et al., NeurIPS 2021, "No Fear of Heterogeneity") cho FLlib.
Dat vao: FLlib/system/flcore/servers/serverccvr.py

Quy trinh: huan luyen FedAvg binh thuong. Moi `eval_gap` vong, TREN BAN SAO cua mo hinh toan cuc:
  1) moi client tinh mean + covariance DAY DU tung lop (dung encoder toan cuc);
  2) server gop theo cong thuc cua bai bao; lay mau M dac trung ao / lop tu N(mu_c, Sigma_c);
  3) huan luyen lai CHI classifier head tren dac trung ao, roi danh gia mo hinh da hieu chinh.
Mo hinh dang huan luyen KHONG bi thay doi (CCVR la buoc hau xu ly).
"""
import copy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from flcore.servers.serveravg import FedAvg
from utils.overlap_utils import OverlapLogger


class FedCCVR(FedAvg):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.ccvr_samples = getattr(args, 'ccvr_samples', 1000)   # M dac trung ao / lop
        self.ccvr_epochs = getattr(args, 'ccvr_epochs', 10)
        self.ccvr_lr = getattr(args, 'ccvr_lr', 0.01)
        self.ccvr_bs = getattr(args, 'ccvr_bs', 256)
        self.ccvr_eps = getattr(args, 'ccvr_eps', 1e-4)           # ridge de Sigma xac dinh duong
        self.num_classes = args.num_classes
        self.overlap_logger = OverlapLogger(args, tag=args.dataset)

    # ---- thong ke lop cua tung client (covariance day du, unbiased) ----
    @torch.no_grad()
    def _client_stats(self, client):
        base = self.global_model.base
        base.eval()
        feats = {c: [] for c in range(self.num_classes)}
        for x, y in client.load_train_data():
            x = x[0] if isinstance(x, list) else x
            z = base(x.to(self.device)).view(x.size(0), -1).float()
            for c in y.unique().tolist():
                feats[int(c)].append(z[y.to(self.device) == c])
        out = {}
        for c, lst in feats.items():
            if lst:
                f = torch.cat(lst, 0)
                n = f.size(0)
                mu = f.mean(0)
                cov = torch.cov(f.T) if n > 1 else torch.zeros(f.size(1), f.size(1), device=f.device)
                out[c] = (n, mu, cov)
        return out

    # ---- gop theo cong thuc CCVR ----
    def _aggregate(self, stats_list):
        gauss = {}
        for c in range(self.num_classes):
            items = [s[c] for s in stats_list if c in s]
            N = sum(i[0] for i in items)
            if N < 2:
                continue
            mu = sum(i[0] * i[1] for i in items) / N
            cov = sum((i[0] - 1) / (N - 1) * i[2] + i[0] / (N - 1) * torch.outer(i[1], i[1])
                      for i in items) - N / (N - 1) * torch.outer(mu, mu)
            d = mu.numel()
            gauss[c] = (mu, 0.5 * (cov + cov.T) + self.ccvr_eps * torch.eye(d, device=cov.device))
        return gauss

    def _calibrated_head(self):
        stats = [self._client_stats(c) for c in self.clients]
        gauss = self._aggregate(stats)
        X, Y = [], []
        for c, (mu, cov) in gauss.items():
            dist = torch.distributions.MultivariateNormal(mu, covariance_matrix=cov)
            X.append(dist.sample((self.ccvr_samples,)))
            Y.append(torch.full((self.ccvr_samples,), c, dtype=torch.int64, device=self.device))
        X, Y = torch.cat(X), torch.cat(Y)
        head = copy.deepcopy(self.global_model.head).to(self.device)
        opt = torch.optim.SGD(head.parameters(), lr=self.ccvr_lr)
        ce = nn.CrossEntropyLoss()
        head.train()
        for _ in range(self.ccvr_epochs):
            for xb, yb in DataLoader(TensorDataset(X, Y), self.ccvr_bs, shuffle=True):
                opt.zero_grad()
                ce(head(xb), yb).backward()
                opt.step()
        return head.state_dict()

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()                       # moi client = mo hinh toan cuc

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model + CCVR calibration")
                orig_head = copy.deepcopy(self.global_model.head.state_dict())
                calib = self._calibrated_head()
                for c in self.clients:               # dung head da hieu chinh de danh gia
                    c.model.head.load_state_dict(calib)
                self.evaluate()
                for c in self.clients:               # tra lai head goc truoc khi huan luyen
                    c.model.head.load_state_dict(orig_head)

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            self.aggregate_parameters()

            # dac trung toan cuc tren du lieu client => Gaussian ma CCVR thuc su dung
            self.overlap_logger.log(
                i, self.selected_clients, self.num_classes,
                test_acc=self.rs_test_acc[-1] if self.rs_test_acc else None,
                encoder=self.global_model.base)

            self.Budget.append(time.time() - s_t)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

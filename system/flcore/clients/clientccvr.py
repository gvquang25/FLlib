from collections import defaultdict
import time
import numpy as np
import torch
from flcore.clients.clientbase import Client


class clientCCVR(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.local_stats = {}

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max(2, max_local_steps // 2))

        for _ in range(max_local_steps):
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.extract_local_stats(trainloader)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    @torch.no_grad()
    def extract_local_stats(self, dataloader=None):
        if dataloader is None:
            from utils.data_utils import read_client_data
            from torch.utils.data import DataLoader
            train_data = read_client_data(self.dataset, self.id, self.args.noniid, self.args.balance, self.args.alpha_dirich, is_train=True, num_clients=self.num_clients)
            dataloader = DataLoader(train_data, batch_size=getattr(self.args, 'test_batch_size', 512), drop_last=False, shuffle=False)
        self.model.eval()
        feats_by_cls = defaultdict(list)
        feat_dim = 512

        for x, y in dataloader:
            x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
            rep = self.model.base(x)
            rep = rep.view(rep.size(0), -1).detach().cpu().float()
            feat_dim = rep.size(1)
            for idx, yy in enumerate(y):
                feats_by_cls[int(yy.item())].append(rep[idx])

        stats = {}
        for c in range(self.num_classes):
            if c in feats_by_cls and len(feats_by_cls[c]) > 0:
                mat = torch.stack(feats_by_cls[c], dim=0)
                n_c = mat.size(0)
                mean_c = mat.mean(dim=0)
                std_c = mat.std(dim=0, unbiased=True) if n_c > 1 else torch.zeros_like(mean_c)
                stats[c] = {'mean': mean_c, 'std': std_c, 'count': int(n_c)}
            else:
                stats[c] = {'mean': torch.zeros(feat_dim), 'std': torch.zeros(feat_dim), 'count': 0}

        self.local_stats = stats
        return stats

from collections import defaultdict
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client


class clientTGP(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.protos = None
        self.global_protos = None
        self.loss_mse = nn.MSELoss()
        self.lamda = getattr(args, 'lamda', 0.1)

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

                rep = self.model.base(x)
                rep_flat = rep.view(rep.size(0), -1)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                if self.global_protos is not None and len(self.global_protos) > 0:
                    proto_new = rep_flat.detach().clone()
                    for idx, yy in enumerate(y):
                        y_c = int(yy.item())
                        if y_c in self.global_protos and self.global_protos[y_c] is not None:
                            proto_new[idx, :] = self.global_protos[y_c].data.to(self.device).view(-1)
                    loss = loss + self.loss_mse(proto_new, rep_flat) * self.lamda

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.collect_protos()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

    def collect_protos(self):
        from utils.data_utils import read_client_data
        from torch.utils.data import DataLoader
        train_data = read_client_data(self.dataset, self.id, self.args.noniid, self.args.balance, self.args.alpha_dirich, is_train=True, num_clients=self.num_clients)
        trainloader = DataLoader(train_data, batch_size=getattr(self.args, 'test_batch_size', 512), drop_last=False, shuffle=False)
        self.model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                rep = self.model.base(x)
                rep_flat = rep.view(rep.size(0), -1)
                for idx, yy in enumerate(y):
                    protos[int(yy.item())].append(rep_flat[idx, :].detach().cpu())

        agg_protos = {}
        stats = {}
        for label, proto_list in protos.items():
            if len(proto_list) > 1:
                mat = torch.stack(proto_list, dim=0)
                agg_protos[label] = mat.mean(dim=0)
                std_c = mat.std(dim=0, unbiased=False)
                std_c = torch.nan_to_num(std_c, nan=0.0, posinf=0.0, neginf=0.0)
                stats[label] = {'mean': agg_protos[label], 'std': std_c, 'count': int(mat.size(0))}
            else:
                agg_protos[label] = proto_list[0]
                stats[label] = {'mean': proto_list[0], 'std': torch.zeros_like(proto_list[0]), 'count': 1}
        self.protos = agg_protos
        self.local_stats = stats

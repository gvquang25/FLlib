"""
servertgp.py -- FedTGP (Zhang et al., AAAI 2024) cho FLlib.
Dat vao: FLlib/system/flcore/servers/servertgp.py
Client dung lai clientProto (giong ban chinh thuc: client = FedProto client).
Chuyen the tu: github.com/TsingZ0/HtFLlib  (servertgp.py), co noi them OverlapLogger.
"""
import time
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from flcore.clients.clientproto import clientProto
from flcore.servers.serverbase import Server
from utils.overlap_utils import OverlapLogger


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientProto)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedTGP server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs                         # khuyen nghi -se 100
        self.margin_threthold = getattr(args, 'margin_threthold', 100.0)

        self.TGP = None            # tao lazily khi biet feature_dim
        self.CEloss = nn.CrossEntropyLoss()
        self.global_protos = None
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        self.min_gap, self.max_gap = None, None
        self.overlap_logger = OverlapLogger(args, tag=args.dataset)

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos()
            self.update_TGP()
            self.send_protos()

            self.overlap_logger.log(
                i, self.selected_clients, self.num_classes,
                mean_override={int(c): p for c, p in self.global_protos.items()},
                test_acc=self.rs_test_acc[-1] if self.rs_test_acc else None)

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])
            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()

    def send_protos(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids, self.uploaded_protos = [], []
        per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for k, v in client.protos.items():
                self.uploaded_protos.append((v, k))
            per_client.append(client.protos)

        # khoang cach toi thieu theo lop giua cac prototype trung binh (de dat margin thich nghi)
        self.gap = torch.ones(self.num_classes, device=self.device) * 1e9
        avg_protos = proto_cluster(per_client)
        for k1 in avg_protos:
            for k2 in avg_protos:
                if k1 > k2:
                    dis = torch.norm(avg_protos[k1] - avg_protos[k2], p=2)
                    self.gap[k1] = torch.min(self.gap[k1], dis)
                    self.gap[k2] = torch.min(self.gap[k2], dis)
        self.min_gap = torch.min(self.gap)
        for c in range(len(self.gap)):
            if self.gap[c] > 1e8:
                self.gap[c] = self.min_gap
        self.max_gap = torch.max(self.gap)

    def update_TGP(self):
        feat_dim = self.uploaded_protos[0][0].numel()
        if self.TGP is None:
            self.TGP = Trainable_Global_Prototypes(
                self.num_classes, feat_dim, feat_dim, self.device).to(self.device)
        TGP_opt = torch.optim.SGD(self.TGP.parameters(), lr=self.server_learning_rate)
        self.TGP.train()
        loss = None
        for _ in range(self.server_epochs):
            loader = DataLoader(self.uploaded_protos, self.batch_size, drop_last=False, shuffle=True)
            for proto, y in loader:
                proto = proto.to(self.device)
                y = torch.as_tensor(y, dtype=torch.int64, device=self.device)
                proto_gen = self.TGP(list(range(self.num_classes)))

                f2 = torch.sum(proto ** 2, 1, keepdim=True)
                c2 = torch.sum(proto_gen ** 2, 1, keepdim=True)
                dist = f2 - 2 * torch.matmul(proto, proto_gen.T) + c2.T
                dist = torch.sqrt(dist.clamp_min(1e-12))

                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                margin = min(self.max_gap.item(), self.margin_threthold)
                dist = dist + one_hot * margin
                loss = self.CEloss(-dist, y)

                TGP_opt.zero_grad()
                loss.backward()
                TGP_opt.step()
        if loss is not None:
            print(f'Server loss: {loss.item()}')
        self.uploaded_protos = []

        self.TGP.eval()
        with torch.no_grad():
            self.global_protos = {c: self.TGP(torch.tensor(c, device=self.device)).detach()
                                  for c in range(self.num_classes)}


def proto_cluster(protos_list):
    clusters = defaultdict(list)
    for protos in protos_list:
        for k, v in protos.items():
            clusters[k].append(v)
    return {k: torch.mean(torch.stack(v), dim=0).detach() for k, v in clusters.items()}


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(nn.Sequential(nn.Linear(feature_dim, server_hidden_dim), nn.ReLU()))
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.as_tensor(class_id, device=self.device)
        return self.fc(self.middle(self.embedings(class_id)))

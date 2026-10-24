import time
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from flcore.clients.clienttgp import clientTGP
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.wandb_utils import log_server_summary


class TrainablePrototypes(nn.Module):
    def __init__(self, num_classes, feature_dim, server_hidden_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        self.middle = nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_ids):
        emb = self.embedings(class_ids)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out


class FedTGP(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientTGP)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedTGP server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = {}

        self.server_learning_rate = getattr(args, 'tgp_server_lr', 0.01)
        self.server_epochs = getattr(args, 'tgp_server_epochs', 100)
        self.margin_threthold = getattr(args, 'margin_threthold', 100.0)
        self.feature_dim = getattr(args, 'z_dim', 512)
        self.server_hidden_dim = self.feature_dim

        self.proto_model = TrainablePrototypes(
            self.num_classes, self.feature_dim, self.server_hidden_dim, self.device
        ).to(self.device)
        self.server_optimizer = torch.optim.SGD(self.proto_model.parameters(), lr=self.server_learning_rate)
        self.CEloss = nn.CrossEntropyLoss()

    def train(self):
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # 1. Clients huấn luyện cục bộ
            for client in self.selected_clients:
                client.train()

            # 2. Thu nhận local prototypes và huấn luyện Trainable Global Prototypes
            self.receive_protos()
            self.update_trainable_protos()
            self.send_protos()

            # 3. Đánh giá mô hình
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / max(len(self.Budget[1:]), 1))

        log_server_summary(self)
        self.save_results()

    def send_protos(self):
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_protos)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_protos_per_class = defaultdict(list)

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            for c, p in client.protos.items():
                self.uploaded_protos.append((p.detach().view(-1), c))
                self.uploaded_protos_per_class[c].append(p.detach().view(-1))

    def update_trainable_protos(self):
        if not self.uploaded_protos:
            return

        # 1. Tính khoảng cách trung bình giữa các tâm lớp để xác định Adaptive Margin
        avg_protos = {}
        for c, plist in self.uploaded_protos_per_class.items():
            avg_protos[c] = torch.stack(plist, dim=0).mean(dim=0)

        max_dist = 0.0
        cls_keys = list(avg_protos.keys())
        if len(cls_keys) >= 2:
            mat = torch.stack([avg_protos[c] for c in cls_keys], dim=0)
            dmat = torch.cdist(mat, mat, p=2)
            max_dist = float(dmat.max().item())
        margin = min(max_dist, float(self.margin_threthold))

        # 2. Tạo DataLoader cho các prototype tải lên
        p_tensors = torch.stack([item[0] for item in self.uploaded_protos], dim=0).to(self.device)
        y_tensors = torch.tensor([item[1] for item in self.uploaded_protos], dtype=torch.long, device=self.device)
        proto_loader = DataLoader(TensorDataset(p_tensors, y_tensors), batch_size=self.batch_size, shuffle=True)

        self.proto_model.train()
        all_class_ids = torch.arange(self.num_classes, device=self.device)

        for _ in range(self.server_epochs):
            for p_batch, y_batch in proto_loader:
                gen_protos = self.proto_model(all_class_ids)  # [C, d]
                # Khoảng cách Euclidean giữa p_batch [B, d] và gen_protos [C, d]
                dist = torch.cdist(p_batch, gen_protos, p=2)  # [B, C]
                # Cộng Adaptive Margin vào đúng cột lớp thực y_batch để ép đẩy xa các lớp khác
                one_hot = F.one_hot(y_batch, num_classes=self.num_classes).float()
                dist_with_margin = dist + one_hot * margin
                loss = self.CEloss(-dist_with_margin, y_batch)

                self.server_optimizer.zero_grad()
                loss.backward()
                self.server_optimizer.step()

        self.proto_model.eval()
        with torch.no_grad():
            final_protos = self.proto_model(all_class_ids).detach().cpu()
            self.global_protos = {c: final_protos[c] for c in range(self.num_classes)}

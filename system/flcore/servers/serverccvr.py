import copy
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

from flcore.clients.clientccvr import clientCCVR
from flcore.servers.serverbase import Server
from flcore.servers.servernew import proto_aggregation
from utils.result_utils import print_experiment_config
from utils.wandb_utils import log_server_summary


class CCVR(Server):
    """
    Server cho CCVR (Classifier Calibration with Virtual Representations - NeurIPS 2021):
    1. Huấn luyện và tổng hợp trọng số theo FedAvg.
    2. Tổng hợp kỳ vọng và phương sai gộp toàn cục (Pooled Mean & Covariance) từ các client.
    3. Lấy mẫu đặc trưng ảo từ phân phối Gaussian toàn cục N(mu_c, Sigma_c) để hiệu chỉnh
       bộ phân loại tuyến tính (Classifier Head) trước khi phát tán về các client.
    """
    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientCCVR)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating CCVR server and clients.")

        self.Budget = []
        self.global_stats = {}
        self.num_virtual_samples = getattr(args, 'ccvr_samples', 100)
        self.calib_epochs = getattr(args, 'ccvr_epochs', 5)

    def train(self):
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            # 1. Clients huấn luyện cục bộ và trích xuất (mean, std, count)
            for client in self.selected_clients:
                client.train()

            # 2. Tổng hợp mô hình toàn cục (FedAvg) + Tổng hợp phân phối Gaussian toàn cục
            self.receive_models()
            self.aggregate_parameters()

            uploaded_stats = [c.local_stats for c in self.selected_clients if getattr(c, 'local_stats', None)]
            self.global_stats = proto_aggregation(uploaded_stats)

            # 3. Hiệu chỉnh Classifier Head bằng Virtual Representations (CCVR Calibration)
            self.calibrate_global_classifier()
            self.send_models()

            # 4. Đánh giá sau mỗi round
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global/calibrated models")
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

    def calibrate_global_classifier(self):
        if not self.global_stats or not hasattr(self.global_model, 'head'):
            return

        virt_feats = []
        virt_labels = []
        for c, s in self.global_stats.items():
            mean_c = s['mean'].to(self.device)
            std_c = s['std'].to(self.device)
            if torch.norm(mean_c) <= 1e-8:
                continue
            eps = torch.randn(self.num_virtual_samples, mean_c.numel(), device=self.device)
            z_virt = torch.relu(mean_c.unsqueeze(0) + eps * std_c.unsqueeze(0))
            virt_feats.append(z_virt.detach())
            virt_labels.append(torch.full((self.num_virtual_samples,), int(c), dtype=torch.long, device=self.device))

        if not virt_feats:
            return

        X_virt = torch.cat(virt_feats, dim=0)
        Y_virt = torch.cat(virt_labels, dim=0)
        loader = DataLoader(TensorDataset(X_virt, Y_virt), batch_size=64, shuffle=True)

        self.global_model.head.train()
        opt_head = torch.optim.SGD(self.global_model.head.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        loss_fn = torch.nn.CrossEntropyLoss()

        for _ in range(self.calib_epochs):
            for bx, by in loader:
                opt_head.zero_grad()
                out = self.global_model.head(bx)
                loss = loss_fn(out, by)
                loss.backward()
                opt_head.step()

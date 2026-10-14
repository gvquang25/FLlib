from collections import defaultdict
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client
from utils.tsne_utils import plot_client_tsne
from utils.wandb_utils import log_client_train


class clientNew(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.args = getattr(self, 'args', args)

        # Thuộc tính FedNew gốc
        self.global_prototypes = {}
        self.global_stds = {}
        self.local_stats = {}
        self.loss_mse = nn.MSELoss()
        self.lamda = getattr(args, 'lamda', 1.0)

        # Tách Backbone (Encoder) và Classification Head để trích xuất rep phục vụ tính mean/std và t-SNE
        # Giữ nguyên self.optimizer mặc định của class cha Client cho toàn bộ model (không tách 2 optimizer)
        self.encoder, self.head = self._split_model()

    def _split_model(self):
        """Tách backbone (trích xuất đặc trưng) và head phân lớp."""
        if hasattr(self.model, 'base') and hasattr(self.model, 'head'):
            return self.model.base, self.model.head
        if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
            return self.model.features, self.model.classifier
        return self.model, nn.Identity()

    def extract_features(self, x):
        """
        Trích xuất đặc trưng và logits:
        - Xử lý đầu vào x (nếu là list ảnh như một số bộ dữ liệu phân đoạn).
        - rep: vector đặc trưng 2D (B, D).
        - out: logits phân lớp. Lưu ý: một số head trong PFLlib nhận tensor đa chiều (B, C, H, W)
          trong khi head dạng Linear nhận tensor 2D phẳng (B, D). Do đó dùng fallback an toàn.
        """
        x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
        rep_raw = self.encoder(x)
        rep = rep_raw.view(rep_raw.size(0), -1)
        try:
            out = self.head(rep_raw)
        except (RuntimeError, ValueError):
            out = self.head(rep)
        return rep, out

    def extract_local_stats(self, dataloader=None):
        """
        Trích xuất mean, std, count theo từng class:
        - Chỉ forward qua encoder, không qua head để tối ưu tốc độ.
        - Xử lý biên: 1 mẫu (std = 1e-6), 0 mẫu (mean/std = 0).
        """
        dataloader = dataloader or self.load_train_data()
        self.model.eval()
        features_per_class = defaultdict(list)
        feat_dim = None

        with torch.no_grad():
            for x, y in dataloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Chỉ forward qua encoder để tiết kiệm FLOPs
                rep_raw = self.encoder(x)
                rep = rep_raw.view(rep_raw.size(0), -1)

                if feat_dim is None and rep.size(0) > 0:
                    feat_dim = rep.size(1)
                assert rep.size(1) == feat_dim, f"Dimension mismatch: {rep.size(1)} vs {feat_dim}"

                for feat, label in zip(rep, y):
                    features_per_class[int(label.item())].append(feat.unsqueeze(0))

        feat_dim = feat_dim or 512
        num_classes = getattr(self, 'num_classes', 10)
        all_classes = set(range(num_classes)).union(features_per_class.keys())

        client_stats = {}
        for c in sorted(all_classes):
            c_int = int(c)
            if c_int in features_per_class and len(features_per_class[c_int]) > 0:
                class_feats = torch.cat(features_per_class[c_int], dim=0)
                n_samples = class_feats.size(0)
                mean_c = torch.mean(class_feats, dim=0)
                std_c = torch.std(class_feats, dim=0, unbiased=False) if n_samples > 1 else torch.full_like(mean_c, 1e-6)
                client_stats[c_int] = {'mean': mean_c.cpu(), 'std': std_c.cpu(), 'count': int(n_samples)}
            else:
                client_stats[c_int] = {'mean': torch.zeros(feat_dim), 'std': torch.zeros(feat_dim), 'count': 0}

        self.local_stats = client_stats
        return client_stats

    def receive_global_stats(self, global_stats):
        """Cập nhật dict mà không làm mất các prototype đã tồn tại trước đó."""
        if not global_stats:
            return
        for c, stats in global_stats.items():
            if stats and isinstance(stats, dict):
                k = int(c)
                if stats.get('mean') is not None:
                    self.global_prototypes[k] = stats['mean'].clone().detach().to(self.device)
                if stats.get('std') is not None:
                    self.global_stds[k] = stats['std'].clone().detach().to(self.device)

    def set_protos(self, global_stats):
        self.receive_global_stats(global_stats)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_steps = np.random.randint(1, max(2, self.local_epochs // 2 + 1)) if self.train_slow else self.local_epochs

        losses = []

        for _ in range(max_local_steps):
            for x, y in trainloader:
                x = x[0].to(self.device) if isinstance(x, list) else x.to(self.device)
                y = y.to(self.device)
                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep, out = self.extract_features(x)
                loss_ce = self.loss(out, y)
                loss_total = loss_ce

                self.optimizer.zero_grad()
                loss_total.backward()
                self.optimizer.step()

                losses.append(loss_total.item())

        # Ghi log WandB an toàn
        log_client_train(self, losses)

        self.extract_local_stats(trainloader)
        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def visualize_tsne(self, round_idx=0, max_samples=1000, save_dir="./tsne_resultsnew"):
        return plot_client_tsne(self, round_idx=round_idx, max_samples=max_samples, save_dir=save_dir)


ClientNew = clientNew
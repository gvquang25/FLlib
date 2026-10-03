from collections import defaultdict
import copy
import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from flcore.clients.clientbase import Client


class clientProtoStd(Client):
    """
    Client for FedProtoStd:
    - Maintains local prototypes (mean feature vectors) and standard deviations (std per feature dimension).
    - Receives global prototypes and stds from the server.
    - Trains using joint objective:
        L_task (CE on real data)
        + L_align (variance-weighted alignment between real features and global prototypes)
        + L_aug (CE on virtual features sampled from N(mu_c, sigma_c^2) to learn missing classes)
    """
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.lamda_align = getattr(args, 'lamda_align', 1.0)
        self.lamda_aug = getattr(args, 'lamda_aug', 1.0)
        self.num_aug_samples = getattr(args, 'num_aug_samples', 10)
        self.eval_mode = getattr(args, 'eval_mode', 'head')

        self.global_protos = None
        self.global_stds = None

        self.local_protos = {}
        self.local_stds = {}
        self.local_counts = {}

    def set_protos_and_stds(self, global_protos, global_stds):
        self.global_protos = copy.deepcopy(global_protos)
        self.global_stds = copy.deepcopy(global_stds)

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()
        self.model.train()

        max_local_steps = self.local_epochs
        if self.train_slow:
            max_local_steps = np.random.randint(1, max(2, max_local_steps // 2))

        for step in range(max_local_steps):
            for i, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss_task = self.loss(output, y)

                loss_align = torch.tensor(0.0, device=self.device)
                loss_aug = torch.tensor(0.0, device=self.device)

                if self.global_protos is not None and self.global_stds is not None:
                    # 1. Distribution-weighted Alignment Loss for Backbone
                    align_losses = []
                    for idx, yy in enumerate(y):
                        c = yy.item()
                        if (c in self.global_protos and self.global_protos[c] is not None and
                                c in self.global_stds and self.global_stds[c] is not None):
                            mu_c = self.global_protos[c].to(self.device)
                            sigma_c = self.global_stds[c].to(self.device)
                            var_c = torch.clamp(sigma_c ** 2, min=1e-4)

                            # Relative weights normalized so mean weight is 1.0
                            inv_var = 1.0 / var_c
                            weights = inv_var / torch.mean(inv_var)

                            diff_sq = (rep[idx] - mu_c) ** 2
                            weighted_diff = torch.mean(weights * diff_sq)
                            align_losses.append(weighted_diff)

                    if len(align_losses) > 0:
                        loss_align = torch.stack(align_losses).mean()

                    # 2. Gaussian Feature Augmentation for Classifier Head
                    aug_feats = []
                    aug_targets = []
                    for c in range(self.num_classes):
                        if (c in self.global_protos and self.global_protos[c] is not None and
                                c in self.global_stds and self.global_stds[c] is not None):
                            mu_c = self.global_protos[c].to(self.device)
                            sigma_c = self.global_stds[c].to(self.device)

                            # Sample virtual features z ~ N(mu_c, sigma_c^2)
                            eps = torch.randn(self.num_aug_samples, mu_c.shape[0], device=self.device)
                            z_tilde = mu_c.unsqueeze(0) + eps * sigma_c.unsqueeze(0)
                            aug_feats.append(z_tilde)
                            aug_targets.append(torch.full((self.num_aug_samples,), c, dtype=torch.long, device=self.device))

                    if len(aug_feats) > 0:
                        all_aug_feats = torch.cat(aug_feats, dim=0)
                        all_aug_targets = torch.cat(aug_targets, dim=0)
                        aug_outputs = self.model.head(all_aug_feats)
                        loss_aug = self.loss(aug_outputs, all_aug_targets)

                total_loss = loss_task + self.lamda_align * loss_align + self.lamda_aug * loss_aug

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Extract empirical prototypes and stds over the full local training set
        self.collect_protos_and_stds()

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def collect_protos_and_stds(self):
        trainloader = self.load_train_data()
        self.model.eval()

        features_by_class = defaultdict(list)
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                for idx, yy in enumerate(y):
                    c = yy.item()
                    features_by_class[c].append(rep[idx].detach())

        self.local_protos = {}
        self.local_stds = {}
        self.local_counts = {}

        for c, feat_list in features_by_class.items():
            if len(feat_list) > 0:
                feats = torch.stack(feat_list, dim=0)  # Shape: (N_c, D)
                n_c = feats.shape[0]
                self.local_counts[c] = n_c

                mean_c = torch.mean(feats, dim=0)
                self.local_protos[c] = mean_c.cpu()

                if n_c > 1:
                    var_c = torch.var(feats, dim=0, unbiased=True)
                else:
                    var_c = torch.full_like(mean_c, 1e-2)

                var_c = torch.clamp(var_c, min=1e-4)
                self.local_stds[c] = torch.sqrt(var_c).cpu()

    def test_metrics(self, model=None):
        testloader = self.load_test_data()
        if model is None:
            model = self.model
        model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = model.base(x)

                if self.eval_mode == 'proto' and self.global_protos is not None and self.global_stds is not None:
                    # Metric classification: Gaussian Mahalanobis NLL distance
                    dist = float('inf') * torch.ones(y.shape[0], self.num_classes, device=self.device)
                    for c in range(self.num_classes):
                        if (c in self.global_protos and self.global_protos[c] is not None and
                                c in self.global_stds and self.global_stds[c] is not None):
                            mu_c = self.global_protos[c].to(self.device)
                            sigma_c = self.global_stds[c].to(self.device)
                            var_c = torch.clamp(sigma_c ** 2, min=1e-6)

                            diff_sq = (rep - mu_c.unsqueeze(0)) ** 2
                            d_c = torch.sum(diff_sq / var_c.unsqueeze(0) + torch.log(var_c.unsqueeze(0)), dim=1)
                            dist[:, c] = d_c

                    pred = torch.argmin(dist, dim=1)
                    test_acc += torch.sum(pred == y).item()
                    output = -dist
                else:
                    output = model.head(rep)
                    pred = torch.argmax(output, dim=1)
                    test_acc += torch.sum(pred == y).item()

                test_num += y.shape[0]
                y_prob.append(output.detach().cpu().numpy())

                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        try:
            auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        except Exception:
            auc = 0.0

        return test_acc, test_num, auc

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.eval()

        train_num = 0
        losses = 0.0
        with torch.no_grad():
            for x, y in trainloader:
                if isinstance(x, list):
                    x = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                rep = self.model.base(x)
                output = self.model.head(rep)
                loss = self.loss(output, y)

                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        return losses, train_num

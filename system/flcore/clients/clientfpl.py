import copy
import time
import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientbase import Client


def agg_func(protos):
    """
    Returns the average of the weights (prototypes) per label.
    """
    for [label, proto_list] in protos.items():
        if len(proto_list) > 1:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:
            protos[label] = proto_list[0]

    return protos


class clientFPL(Client):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)

        self.infoNCET = getattr(args, 'infoNCET', 0.1)
        self.protos = None
        self.global_protos = None

    def calculate_infonce(self, f_now, f_pos, f_neg):
        f_proto = torch.cat((f_pos, f_neg), dim=0)
        l = torch.cosine_similarity(f_now, f_proto, dim=1)
        l = l / self.infoNCET

        exp_l = torch.exp(l).view(1, -1)
        pos_mask = torch.tensor(
            [1] * f_pos.shape[0] + [0] * f_neg.shape[0],
            dtype=torch.float
        ).to(self.device).view(1, -1)

        pos_l = exp_l * pos_mask
        infonce_loss = -torch.log(pos_l.sum(1) / exp_l.sum(1))
        return infonce_loss

    def hierarchical_info_loss(self, f_now, label, all_f, mean_f, all_global_protos_keys):
        label_val = label.item()
        pos_mask = (all_global_protos_keys == label_val)
        pos_indices = np.where(pos_mask)[0]
        pos_idx = pos_indices[0]

        f_pos = all_f[pos_idx].to(self.device)

        neg_indices = np.where(~pos_mask)[0]
        if len(neg_indices) > 0:
            f_neg = torch.cat([all_f[i] for i in neg_indices], dim=0).to(self.device)
        else:
            f_neg = torch.empty((0, f_pos.shape[1]), device=self.device)

        xi_info_loss = self.calculate_infonce(f_now, f_pos, f_neg)

        mean_f_pos = mean_f[pos_idx].to(self.device).view(1, -1)
        cu_info_loss = nn.MSELoss()(f_now, mean_f_pos)

        return xi_info_loss + cu_info_loss

    def train(self):
        trainloader = self.load_train_data()
        start_time = time.time()

        self.model.to(self.device)
        self.model.train()

        # 1.3 Tiền xử lý global prototypes (MỘT LẦN trước vòng epoch, khi self.global_protos khác None và không rỗng)
        has_global_protos = (self.global_protos is not None) and (len(self.global_protos) > 0)
        if has_global_protos:
            all_global_protos_keys = np.array(list(self.global_protos.keys()))
            all_f = []
            mean_f = []
            for protos_key in all_global_protos_keys:
                temp_f = torch.cat(self.global_protos[protos_key], dim=0).to(self.device)
                all_f.append(temp_f.detach().cpu())
                mean_f.append(torch.mean(temp_f, dim=0).detach().cpu())

        max_local_epochs = self.local_epochs
        if self.train_slow:
            max_local_epochs = np.random.randint(1, max(2, max_local_epochs // 2 + 1))

        agg_protos_label = {}
        for epoch in range(max_local_epochs):
            agg_protos_label = {}
            for i_batch, (x, y) in enumerate(trainloader):
                if isinstance(x, list):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)

                if self.train_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))

                # Trích representation f và outputs
                if hasattr(self.model, 'features') and hasattr(self.model, 'classifier'):
                    f = self.model.features(x)
                    if f.dim() > 2:
                        f = torch.flatten(f, start_dim=1)
                    outputs = self.model.classifier(f)
                elif hasattr(self.model, 'base') and hasattr(self.model, 'head'):
                    f = self.model.base(x)
                    if f.dim() > 2:
                        f = torch.flatten(f, start_dim=1)
                    outputs = self.model.head(f)
                else:
                    f = self.model(x)
                    if f.dim() > 2:
                        f = torch.flatten(f, start_dim=1)
                    outputs = f

                if f.dim() > 2:
                    f = torch.flatten(f, start_dim=1)

                lossCE = self.loss(outputs, y)

                if not has_global_protos:
                    loss_InfoNCE = 0 * lossCE
                else:
                    i_total = 0
                    loss_InfoNCE = None

                    for label in y:
                        if label.item() in self.global_protos.keys():
                            f_now = f[i_total].unsqueeze(0)
                            loss_instance = self.hierarchical_info_loss(
                                f_now, label, all_f, mean_f, all_global_protos_keys
                            )

                            if loss_InfoNCE is None:
                                loss_InfoNCE = loss_instance
                            else:
                                loss_InfoNCE = loss_InfoNCE + loss_instance
                        i_total += 1

                    if loss_InfoNCE is None:
                        loss_InfoNCE = 0 * lossCE
                    else:
                        loss_InfoNCE = loss_InfoNCE / i_total

                loss = lossCE + loss_InfoNCE

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Ở epoch cuối: gom prototype theo nhãn
                if epoch == max_local_epochs - 1:
                    for i_sample in range(len(y)):
                        label_val = y[i_sample].item()
                        if label_val in agg_protos_label:
                            agg_protos_label[label_val].append(f[i_sample, :].detach())
                        else:
                            agg_protos_label[label_val] = [f[i_sample, :].detach()]

        self.protos = agg_func(agg_protos_label)

        if self.learning_rate_decay:
            self.learning_rate_scheduler.step()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time

    def set_protos(self, global_protos):
        self.global_protos = copy.deepcopy(global_protos)

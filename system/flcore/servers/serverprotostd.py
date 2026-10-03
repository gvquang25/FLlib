import time
import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from flcore.clients.clientprotostd import clientProtoStd
from flcore.servers.serverbase import Server


class FedProtoStd(Server):
    """
    Server for FedProtoStd:
    - Coordinates communication of Prototypes (mean vectors) and Standard Deviations (stds).
    - Aggregates uploaded prototypes using sample-weighted averaging.
    - Aggregates variances using the Law of Total Variance:
        Var(Z | Y=c) = E[Var(Z | k, Y=c)] + Var(E[Z | k, Y=c])
                     = sum_k (N_kc / N_c) * [ sigma_kc^2 + (mu_kc - mu_c)^2 ]
    - Broadcasts global prototypes and stds to all clients.
    - Logs performance metrics to WandB and TensorBoard if --log is set.
    """
    def __init__(self, args, times):
        super().__init__(args, times)

        self.set_slow_clients()
        self.set_clients(clientProtoStd)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.global_protos = {c: None for c in range(args.num_classes)}
        self.global_stds = {c: None for c in range(args.num_classes)}

    def train(self):
        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_protos_and_stds()
            self.aggregate_protos_and_stds()
            self.send_protos_and_stds()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, f"Round {i} cost: {self.Budget[-1]:.2f}s")

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        if len(self.rs_test_acc) > 0:
            print(f"{max(self.rs_test_acc):.4f}")
        if len(self.Budget) > 1:
            print(f"Average round time: {sum(self.Budget[1:]) / len(self.Budget[1:]):.2f}s")

        self.save_results()

        if getattr(self.args, 'log', False) and wandb is not None:
            wandb.finish()

    def send_protos_and_stds(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos_and_stds(self.global_protos, self.global_stds)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos_and_stds(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_protos = []
        self.uploaded_stds = []
        self.uploaded_counts = []

        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.local_protos)
            self.uploaded_stds.append(client.local_stds)
            self.uploaded_counts.append(client.local_counts)

    def aggregate_protos_and_stds(self):
        for c in range(self.num_classes):
            total_samples = 0
            weighted_proto = None

            # 1. Compute Global Prototype (weighted mean by sample counts)
            for p_dict, c_dict in zip(self.uploaded_protos, self.uploaded_counts):
                if c in p_dict and p_dict[c] is not None and c in c_dict and c_dict[c] > 0:
                    count = c_dict[c]
                    proto = p_dict[c].to(self.device)
                    if weighted_proto is None:
                        weighted_proto = proto * count
                    else:
                        weighted_proto += proto * count
                    total_samples += count

            if total_samples > 0:
                global_c_proto = weighted_proto / total_samples
                self.global_protos[c] = global_c_proto.cpu()
            else:
                continue

            # 2. Compute Global Variance via Law of Total Variance:
            # Var(Z | Y=c) = sum_k (N_kc / N_c) * [ sigma_kc^2 + (mu_kc - mu_c)^2 ]
            pooled_var = torch.zeros_like(global_c_proto)
            for p_dict, s_dict, c_dict in zip(self.uploaded_protos, self.uploaded_stds, self.uploaded_counts):
                if (c in p_dict and p_dict[c] is not None and
                        c in s_dict and s_dict[c] is not None and
                        c in c_dict and c_dict[c] > 0):
                    count = c_dict[c]
                    weight = float(count) / float(total_samples)

                    local_p = p_dict[c].to(self.device)
                    local_s = s_dict[c].to(self.device)

                    intra_var = local_s ** 2
                    drift_var = (local_p - global_c_proto) ** 2
                    pooled_var += weight * (intra_var + drift_var)

            pooled_var = torch.clamp(pooled_var, min=1e-6)
            self.global_stds[c] = torch.sqrt(pooled_var).cpu()

    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        tot_samples = sum(stats[1])
        if tot_samples > 0:
            test_acc = sum(stats[2]) * 1.0 / tot_samples
            test_auc = sum(stats[3]) * 1.0 / tot_samples
            accs = [a / n for a, n in zip(stats[2], stats[1]) if n > 0]
        else:
            test_acc, test_auc, accs = 0.0, 0.0, [0.0]

        tot_train_samples = sum(stats_train[1])
        if tot_train_samples > 0:
            train_loss = sum(stats_train[2]) * 1.0 / tot_train_samples
        else:
            train_loss = 0.0

        if acc is None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)

        self.rs_test_auc.append(test_auc)

        if loss is None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        test_acc_std = np.std(accs) if len(accs) > 0 else 0.0

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        print("Std Test Accuracy: {:.4f}".format(test_acc_std))

        # Log metrics to TensorBoard and WandB
        if getattr(self.args, 'log', False):
            if hasattr(self, 'writer') and self.writer is not None:
                self.writer.add_scalar("charts/train_loss", train_loss, self.current_round)
                self.writer.add_scalar("charts/test_acc", test_acc, self.current_round)
                self.writer.add_scalar("charts/test_auc", test_auc, self.current_round)
                self.writer.add_scalar("charts/test_acc_std", test_acc_std, self.current_round)

            if wandb is not None:
                wandb.log({
                    "charts/train_loss": train_loss,
                    "charts/test_acc": test_acc,
                    "charts/test_auc": test_auc,
                    "charts/test_acc_std": test_acc_std,
                }, step=self.current_round)

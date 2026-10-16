import time
import copy
import numpy as np
import torch
import torch.nn as nn
from flcore.clients.clientfpl import clientFPL
from flcore.servers.serverbase import Server
from utils.finch import FINCH
try:
    import wandb
    if not hasattr(wandb, 'init'):
        wandb = None
except (ImportError, Exception):
    wandb = None


def proto_aggregation(uploaded_protos):
    """
    Hierarchical prototype aggregation using FINCH clustering per class.
    """
    agg_protos_label = dict()
    protos_iterable = uploaded_protos.values() if isinstance(uploaded_protos, dict) else uploaded_protos
    for local_protos in protos_iterable:
        if local_protos is None:
            continue
        for label in local_protos.keys():
            if label in agg_protos_label:
                agg_protos_label[label].append(local_protos[label])
            else:
                agg_protos_label[label] = [local_protos[label]]

    for label, proto_list in list(agg_protos_label.items()):
        if len(proto_list) > 1:
            proto_arr = [p.squeeze().detach().cpu().numpy().reshape(-1) for p in proto_list]
            proto_arr = np.array(proto_arr)

            try:
                c, num_clust, req_c = FINCH(
                    proto_arr,
                    initial_rank=None,
                    req_clust=None,
                    distance='cosine',
                    ensure_early_exit=False,
                    verbose=False
                )
                if c.ndim == 1:
                    class_cluster_array = c
                elif c.ndim == 2 and c.shape[1] > 0:
                    class_cluster_array = c[:, -1]
                else:
                    class_cluster_array = np.zeros(len(proto_list), dtype=int)
            except Exception:
                class_cluster_array = np.zeros(len(proto_list), dtype=int)

            unique_clusters = np.unique(class_cluster_array)
            agg_selected_proto = []

            for cluster_index in unique_clusters:
                selected_proto_list = proto_arr[class_cluster_array == cluster_index]
                proto = np.mean(selected_proto_list, axis=0, keepdims=True)
                agg_selected_proto.append(torch.tensor(proto, dtype=torch.float32))

            agg_protos_label[label] = agg_selected_proto
        elif len(proto_list) == 1:
            p = proto_list[0].data.detach().cpu().float()
            if p.dim() == 1:
                p = p.unsqueeze(0)
            elif p.dim() > 2:
                p = p.view(1, -1)
            agg_protos_label[label] = [p]
        else:
            agg_protos_label.pop(label, None)

    return agg_protos_label


class serverFPL(Server):
    def __init__(self, args, times):
        super().__init__(args, times)

        # select slow clients
        self.set_slow_clients()
        self.set_clients(clientFPL)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.global_protos = {}
        self.uploaded_protos = []

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.current_round = i
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_protos()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and i % self.dlg_gap == 0:
                self.call_dlg(i)
            self.aggregate_parameters()

            self.receive_protos()
            self.global_protos = proto_aggregation(self.uploaded_protos)

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        if len(self.Budget) > 1:
            print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        self.save_results()
        self.save_global_model()

        if self.args.log:
            self.writer.flush()
            if wandb is not None:
                wandb.finish()

        if self.num_new_clients > 0:
            self.eval_new_clients = True
            self.set_new_clients(clientFPL)
            print(f"\n-------------Fine tuning round-------------")
            print("\nEvaluate new clients")
            self.evaluate()

    def send_protos(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()

            client.set_protos(self.global_protos)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_ids = []
        self.uploaded_protos = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_protos.append(client.protos)

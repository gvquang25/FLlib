import time
import torch

from flcore.clients.clientnew_proto import clientNewProto
from flcore.servers.serverbase import Server
from utils.wandb_utils import log_server_summary

ServerBase = Server


class FedNewProto(Server):
    """
    Server cho FedNewProto:
    Kế thừa Server (ServerBase), khởi tạo client là clientNewProto.
    Duy trì 100% thuật toán tổng hợp thống kê Weighted Mean & Pooled Variance của FedNew,
    kèm theo bước trực quan hóa t-SNE theo chu kỳ tsne_gap.
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        if not hasattr(self, 'current_round'):
            self.current_round = 0

        self.set_slow_clients()
        self.set_clients(clientNewProto)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.global_stats = {}

    def train(self):
        for i in range(self.global_rounds + 1):
            s_t = time.time()
            self.selected_clients = self.select_clients()

            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")

                tsne_gap = getattr(self.args, 'tsne_gap', None)
                tsne_gap = self.eval_gap if (tsne_gap is None or tsne_gap <= 0) else tsne_gap
                if i % tsne_gap == 0 and len(self.clients) > 0:
                    # Dùng self.clients[0] cố định để theo dõi sự tiến hóa cụm đặc trưng.
                    # Khi join_ratio < 1, client 0 có thể không được chọn train ở round trước;
                    # thông tin round tương ứng đã được thể hiện rõ trên tiêu đề ảnh t-SNE.
                    self.clients[0].visualize_tsne(round_idx=i)

                self.evaluate()

            # 1. Nhận stats từ clients
            self.receive_protos()

            # 2. Server tổng hợp Weighted Mean & Pooled Variance
            self.global_stats = proto_aggregation(self.uploaded_stats)

            # 3. Phân phối global stats xuống clients
            self.send_protos()

            # 4. Client huấn luyện
            for client in self.selected_clients:
                client.train()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))

        log_server_summary(self)

        self.save_results()

    def send_protos(self):
        assert len(self.clients) > 0
        for client in self.clients:
            start_time = time.time()
            client.set_protos(self.global_stats)
            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_protos(self):
        assert len(self.selected_clients) > 0
        self.uploaded_ids = []
        self.uploaded_stats = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            if hasattr(client, 'local_stats') and client.local_stats:
                self.uploaded_stats.append(client.local_stats)
            elif hasattr(client, 'extract_local_stats'):
                self.uploaded_stats.append(client.extract_local_stats())


    def proto_aggregation(self, local_stats_dict=None):
        if local_stats_dict is None:
            self.receive_protos()
            local_stats_dict = self.uploaded_stats
        self.global_stats = proto_aggregation(local_stats_dict)
        return self.global_stats

    aggregate_prototypes_and_stds = proto_aggregation


def proto_aggregation(local_stats_list):
    """
    Weighted Mean & Pooled Variance cho FedNew:
    global_mean = sum(n_k * mean_k) / sum(n_k)
    pooled_var = sum(n_k * (std_k^2 + (mean_k - global_mean)^2)) / sum(n_k)
    """
    if isinstance(local_stats_list, dict):
        local_stats_list = list(local_stats_list.values())

    if not local_stats_list:
        return {}

    all_classes = set()
    feat_dim = None

    for stats in local_stats_list:
        all_classes.update(stats.keys())
        if feat_dim is None:
            for s in stats.values():
                if isinstance(s, dict) and 'mean' in s and isinstance(s['mean'], torch.Tensor):
                    feat_dim = s['mean'].size(0)
                    break

    feat_dim = feat_dim or 512
    global_stats = {}

    for c in sorted(all_classes):
        valid_stats = [s[c] for s in local_stats_list if c in s and s[c].get('count', 0) > 0]
        total_count = sum(s['count'] for s in valid_stats)

        if total_count == 0:
            global_stats[c] = {'mean': torch.zeros(feat_dim), 'std': torch.zeros(feat_dim)}
            continue

        global_mean = torch.zeros(feat_dim, dtype=torch.float32)
        for s in valid_stats:
            global_mean += s['count'] * s['mean'].float()
        global_mean /= total_count

        pooled_var = torch.zeros(feat_dim, dtype=torch.float32)
        for s in valid_stats:
            n_k = s['count']
            local_var = (s['std'].float()) ** 2
            mean_dev_sq = (s['mean'].float() - global_mean) ** 2
            pooled_var += n_k * (local_var + mean_dev_sq)

        pooled_var /= total_count
        global_std = torch.sqrt(torch.clamp(pooled_var, min=1e-8))

        global_stats[c] = {'mean': global_mean, 'std': global_std}

    return global_stats


aggregate_prototypes_and_stds = proto_aggregation
ServerNewProto = FedNewProto
serverNewProto = FedNewProto

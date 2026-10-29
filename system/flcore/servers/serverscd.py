import time
import numpy as np
import torch

from flcore.clients.clientscd import clientSCD
from flcore.servers.serverbase import Server
from utils.result_utils import print_experiment_config
from utils.wandb_utils import log_server_summary


class FedSCD(Server):
    """
    Server cho FedSCD:
    - Kế thừa trực tiếp từ Server (serverbase), hoạt động hoàn toàn độc lập.
    - Khởi tạo client là clientSCD thông qua self.set_clients(clientSCD).
    - Triển khai E-Step phân tán:
      1. Weighted Mean & Pooled Variance từ local_stats của các client.
      2. Momentum Update (EMA) cho Prototype và Variance kế thừa từ MoCo/PCL.
      3. PCL/SCD Concentration Estimation (ước lượng nhiệt độ động phi cho từng cụm lớp).
    """

    def __init__(self, args, times):
        super().__init__(args, times)

        if not hasattr(self, 'current_round'):
            self.current_round = 0

        self.set_slow_clients()
        self.set_clients(clientSCD)

        # Tham số Momentum EMA và nhiệt độ cơ sở
        self.pcl_momentum = getattr(args, 'proto_momentum', 0.1)
        self.base_temperature = getattr(args, 'infoNCET', 0.1)
        self.phi_range = getattr(args, 'phi_range', 0.1)

        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedSCD server and clients.")

        self.Budget = []
        self.num_classes = args.num_classes
        self.global_stats = {}

    def train(self):
        # In cấu hình thực nghiệm và thiết lập Non-IID
        print_experiment_config(self.args, self.times)

        for i in range(self.global_rounds + 1):
            self.current_round = i
            s_t = time.time()
            self.selected_clients = self.select_clients()

            # 1. Clients huấn luyện cục bộ (M-Step)
            for client in self.selected_clients:
                client.train()

            # 2. Nhận stats từ clients vừa huấn luyện
            self.receive_protos()

            # 3. Server tổng hợp E-Step phân tán
            self.global_stats = proto_aggregation_pcl(
                self.uploaded_stats,
                prev_global_stats=self.global_stats,
                momentum=self.pcl_momentum,
                base_tau=self.base_temperature,
                phi_range=self.phi_range
            )

            # 4. Phân phối global stats (mean, std, phi, count) xuống tất cả clients
            self.send_protos()

            # 5. Đánh giá và trực quan hóa sau khi hoàn tất vòng huấn luyện
            if i % self.eval_gap == 0:
                print(f"\n-------------Round number: {i}-------------")
                print("\nEvaluate personalized models")

                tsne_gap = getattr(self.args, 'tsne_gap', None)
                tsne_gap = self.eval_gap if (tsne_gap is None or tsne_gap <= 0) else tsne_gap
                if i % tsne_gap == 0 and len(self.clients) > 0:
                    # Trực quan hóa t-SNE trên client 0 lưu vào ./tsne_resultsscd
                    self.clients[0].visualize_tsne(round_idx=i)

                self.evaluate()

            self.Budget.append(time.time() - s_t)
            print('-' * 50, self.Budget[-1])

            if self.auto_break and self.check_done(acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        # Đảm bảo mô hình sau round cuối luôn được evaluate nếu chưa được đánh giá
        if (self.global_rounds % self.eval_gap != 0):
            self.current_round = self.global_rounds
            print(f"\n-------------Final Evaluation at Round: {self.global_rounds}-------------")
            self.evaluate()

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
        self.global_stats = proto_aggregation_pcl(
            local_stats_dict,
            prev_global_stats=self.global_stats,
            momentum=self.pcl_momentum,
            base_tau=self.base_temperature,
            phi_range=self.phi_range
        )
        return self.global_stats

    aggregate_prototypes_and_stds = proto_aggregation


def proto_aggregation_pcl(local_stats_list, prev_global_stats=None, momentum=0.1, base_tau=0.1, phi_range=0.1):
    """
    Thuật toán E-Step phân tán tổng hợp Prototype và ước lượng nồng độ cụm theo PCL/SCD:
    1. Weighted Mean & Pooled Variance từ local_stats của các client tải lên:
       batch_mean = sum(n_k * mean_k) / sum(n_k)
       pooled_var = sum(n_k * (std_k^2 + (mean_k - batch_mean)^2)) / sum(n_k)
       batch_std  = sqrt(clamp(pooled_var, min=1e-8))
    2. Momentum Prototype & Variance Update (EMA với hệ số cập nhật mẫu mới momentum):
       final_mean = (1.0 - momentum) * prev_mean + momentum * batch_mean
       final_std  = (1.0 - momentum) * prev_std  + momentum * batch_std
       Bảo toàn thống kê của các lớp không có mẫu trong round (batch_count == 0) từ prev_global_stats.
    3. PCL/SCD Concentration Estimation phi:
       raw_phi_c = torch.norm(final_std, p=2) / log(total_count + 10.0)
       scaled_phi = base_tau * (raw_phi_c / max(mean_raw_phi, 1e-6))
       phi_c = clip(scaled_phi, (1.0 - phi_range) * base_tau, (1.0 + phi_range) * base_tau)
       global_stats[c] = {'mean': final_mean, 'std': final_std, 'phi': phi_c, 'count': total_count}
    """
    if isinstance(local_stats_list, dict):
        local_stats_list = list(local_stats_list.values())

    if not local_stats_list and not prev_global_stats:
        return {}

    all_classes = set()
    feat_dim = None

    if local_stats_list:
        for stats in local_stats_list:
            all_classes.update(stats.keys())
            if feat_dim is None:
                for s in stats.values():
                    if isinstance(s, dict) and 'mean' in s and isinstance(s['mean'], torch.Tensor):
                        feat_dim = s['mean'].size(0)
                        break

    if prev_global_stats:
        all_classes.update(prev_global_stats.keys())
        if feat_dim is None:
            for s in prev_global_stats.values():
                if isinstance(s, dict) and 'mean' in s and isinstance(s['mean'], torch.Tensor):
                    feat_dim = s['mean'].size(0)
                    break

    feat_dim = feat_dim or 512
    global_stats = {}
    raw_phi_dict = {}

    for c in sorted(all_classes):
        valid_stats = [s[c] for s in local_stats_list if c in s and s[c].get('count', 0) > 0] if local_stats_list else []
        batch_count = sum(s['count'] for s in valid_stats)

        has_prev = (
            prev_global_stats is not None
            and c in prev_global_stats
            and isinstance(prev_global_stats[c], dict)
            and prev_global_stats[c].get('count', 0) > 0
        )

        if batch_count > 0:
            # 1. Tính Weighted Mean và Pooled Variance cho batch hiện tại
            batch_mean = torch.zeros(feat_dim, dtype=torch.float32)
            for s in valid_stats:
                batch_mean += s['count'] * s['mean'].float().cpu()
            batch_mean /= batch_count

            pooled_var = torch.zeros(feat_dim, dtype=torch.float32)
            for s in valid_stats:
                n_k = s['count']
                local_var = (s['std'].float().cpu()) ** 2
                mean_dev_sq = (s['mean'].float().cpu() - batch_mean) ** 2
                pooled_var += n_k * (local_var + mean_dev_sq)

            pooled_var /= batch_count
            batch_std = torch.sqrt(torch.clamp(pooled_var, min=1e-8))

            # 2. Cập nhật Momentum EMA qua các rounds (bao gồm cả count để ổn định mật độ)
            if has_prev:
                prev_mean = prev_global_stats[c]['mean'].float().cpu()
                prev_std = prev_global_stats[c]['std'].float().cpu()
                prev_count = float(prev_global_stats[c].get('count', batch_count))
                final_mean = (1.0 - momentum) * prev_mean + momentum * batch_mean
                final_std = (1.0 - momentum) * prev_std + momentum * batch_std
                total_count = int(round((1.0 - momentum) * prev_count + momentum * batch_count))
            else:
                final_mean = batch_mean
                final_std = batch_std
                total_count = batch_count

            # 3. Ước lượng độ tập trung góc tương đối raw_phi_c chuẩn mặt cầu
            mean_norm = float(torch.norm(final_mean, p=2).item()) + 1e-6
            std_norm = float(torch.norm(final_std, p=2).item()) / mean_norm
            raw_phi_c = std_norm / np.log(max(total_count, 1) + 10.0)
            raw_phi_dict[c] = raw_phi_c
            global_stats[c] = {
                'mean': final_mean,
                'std': final_std,
                'count': total_count
            }

        elif has_prev:
            # Bảo toàn thống kê từ round trước nếu lớp c không xuất hiện trong round này
            final_mean = prev_global_stats[c]['mean'].clone().detach().cpu()
            final_std = prev_global_stats[c]['std'].clone().detach().cpu()
            total_count = prev_global_stats[c].get('count', 0)
            mean_norm = float(torch.norm(final_mean, p=2).item()) + 1e-6
            std_norm = float(torch.norm(final_std, p=2).item()) / mean_norm
            raw_phi_c = std_norm / np.log(max(total_count, 1) + 10.0)
            raw_phi_dict[c] = raw_phi_c
            global_stats[c] = {
                'mean': final_mean,
                'std': final_std,
                'count': total_count
            }
        else:
            # Lớp chưa từng có mẫu xuất hiện
            final_mean = torch.zeros(feat_dim, dtype=torch.float32)
            final_std = torch.zeros(feat_dim, dtype=torch.float32)
            raw_phi_dict[c] = None
            global_stats[c] = {
                'mean': final_mean,
                'std': final_std,
                'phi': float(base_tau),
                'count': 0
            }

    # 4. Chuẩn hóa raw_phi trên toàn bộ các lớp có mặt sao cho trung bình bằng base_tau
    present_raw_phis = [v for v in raw_phi_dict.values() if v is not None and v > 0]
    if present_raw_phis:
        mean_raw_phi = float(np.mean(present_raw_phis))
        for c in global_stats:
            if raw_phi_dict.get(c) is not None:
                scaled_phi = base_tau * (raw_phi_dict[c] / max(mean_raw_phi, 1e-6))
                phi_c = float(np.clip(scaled_phi, (1.0 - phi_range) * base_tau, (1.0 + phi_range) * base_tau))
                global_stats[c]['phi'] = phi_c
            else:
                global_stats[c]['phi'] = float(base_tau)
    else:
        for c in global_stats:
            global_stats[c]['phi'] = float(base_tau)

    return global_stats


proto_aggregation_scd = proto_aggregation_pcl
ServerSCD = FedSCD
serverSCD = FedSCD
FedNewPCL = FedSCD
ServerNewPCL = FedSCD
serverNewPCL = FedSCD

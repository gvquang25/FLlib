import os
import random
import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from flcore.clients.clientscd import clientSCD
from utils.tsne_utils import plot_client_tsne


class DummyModel(nn.Module):
    def __init__(self, in_dim=16, feat_dim=32, num_classes=5):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU()
        )
        self.head = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.head(self.base(x))


class DummyArgs:
    def __init__(self):
        self.model = DummyModel()
        self.algorithm = "FedSCD"
        self.dataset = "dummy"
        self.device = "cpu"
        self.save_folder_name = "test_run"
        self.num_clients = 1
        self.num_classes = 5
        self.batch_size = 16
        self.local_learning_rate = 0.01
        self.local_epochs = 1
        self.learning_rate_decay = False
        self.learning_rate_decay_gamma = 0.99
        self.privacy = False
        self.dp_sigma = 0.0
        self.log = False


class TestTSNEUtils(unittest.TestCase):
    def setUp(self):
        self.args = DummyArgs()
        self.client = clientSCD(self.args, 0, 100, 100, train_slow=False, send_slow=False)

        # Tạo dữ liệu giả lập 5 cụm (chỉ có lớp 0, 1, 2 trong dữ liệu test để mô phỏng Non-IID)
        torch.manual_seed(42)
        X_list, y_list = [], []
        # Lớp 0, 1, 2 có mặt tại client
        for cls in [0, 1, 2]:
            center = torch.zeros(16)
            center[cls] = 10.0
            x_cls = center + torch.randn(30, 16) * 0.4
            X_list.append(x_cls)
            y_list.append(torch.full((30,), cls, dtype=torch.long))

        self.X = torch.cat(X_list, dim=0)
        self.y = torch.cat(y_list, dim=0)
        testloader = DataLoader(TensorDataset(self.X, self.y), batch_size=30, shuffle=False)
        self.client.load_test_data = lambda: testloader

        # Tạo global prototypes cho cả 5 lớp (0..4)
        # Cố ý làm P_3 và P_4 rất gần nhau để kiểm tra text offset repulsion
        self.client.global_prototypes = {}
        with torch.no_grad():
            for cls in range(5):
                p_vec = torch.zeros(32)
                p_vec[cls] = 5.0
                if cls == 4:
                    # Đặt P_4 gần P_3
                    p_vec = torch.zeros(32)
                    p_vec[3] = 4.98
                    p_vec[4] = 0.05
                self.client.global_prototypes[cls] = p_vec

    def test_rng_preservation(self):
        """Kiểm tra gọi visualize_tsne không làm thay đổi RNG states của python, numpy, torch."""
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)

        py_state_before = random.getstate()
        np_state_before = np.random.get_state()
        torch_state_before = torch.get_rng_state()
        cuda_state_before = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        save_path = self.client.visualize_tsne(round_idx=99, save_dir="./test_tsne_results")
        self.assertIsNotNone(save_path)
        self.assertTrue(os.path.exists(save_path))

        py_state_after = random.getstate()
        np_state_after = np.random.get_state()
        torch_state_after = torch.get_rng_state()
        cuda_state_after = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None

        self.assertEqual(py_state_before, py_state_after, "Python random state must be 100% preserved")
        self.assertTrue(np.array_equal(np_state_before[1], np_state_after[1]), "NumPy RNG state must be 100% preserved")
        self.assertTrue(torch.equal(torch_state_before, torch_state_after), "Torch RNG state must be 100% preserved")
        if cuda_state_before is not None:
            for s_b, s_a in zip(cuda_state_before, cuda_state_after):
                self.assertTrue(torch.equal(s_b, s_a), "CUDA RNG state must be 100% preserved")

        # Kiểm tra chuỗi sinh số ngẫu nhiên sau khi gọi hàm tiếp tục đúng như mong đợi
        val_torch = torch.randn(1).item()
        val_np = np.random.rand()
        val_py = random.random()

        # Reset lại seed ban đầu và sinh tương tự để đối chiếu
        random.seed(12345)
        np.random.seed(12345)
        torch.manual_seed(12345)
        self.assertEqual(val_torch, torch.randn(1).item())
        self.assertEqual(val_np, np.random.rand())
        self.assertEqual(val_py, random.random())

    def test_model_weights_and_train_state_preservation(self):
        """Kiểm tra trọng số model và trạng thái model.training được bảo toàn tuyệt đối 100%."""
        self.client.model.train()
        params_before = [p.clone().detach() for p in self.client.model.parameters()]

        self.client.visualize_tsne(round_idx=10, save_dir="./test_tsne_results")

        self.assertTrue(self.client.model.training, "Trạng thái model.training phải được trả lại ban đầu")
        params_after = list(self.client.model.parameters())

        self.assertEqual(len(params_before), len(params_after))
        for p_b, p_a in zip(params_before, params_after):
            self.assertTrue(torch.equal(p_b, p_a), "Model parameters must remain 100% unchanged")

    def test_centered_hypersphere_silhouette_score_gain(self):
        """Kiểm tra Silhouette score tính trên Centered Hypersphere Projection cao vượt trội so với raw ReLU."""
        from sklearn.metrics import silhouette_score
        with torch.no_grad():
            raw_feats = self.client.encoder(self.X).numpy()
            y_np = self.y.numpy()
            raw_sil = silhouette_score(raw_feats, y_np)

            # Centered Hypersphere Projection
            protos_tensor = torch.stack([self.client.global_prototypes[c] for c in range(5)], dim=0)
            proto_center = torch.mean(protos_tensor, dim=0, keepdim=True)
            feats_proj = torch.nn.functional.normalize(self.client.encoder(self.X) - proto_center, dim=1).numpy()
            proj_sil = silhouette_score(feats_proj, y_np, metric='cosine')

            print(f"\n[Test] Silhouette Raw: {raw_sil:.4f} vs Centered Hypersphere (Cosine): {proj_sil:.4f}")
            self.assertGreater(proj_sil, raw_sil, "Centered Hypersphere Silhouette score must be higher than raw ReLU")


if __name__ == '__main__':
    unittest.main()

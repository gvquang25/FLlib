# Hướng Dẫn Sử Dụng Thuật Toán FedProtoStd

Tài liệu hướng dẫn chi tiết cách cấu hình, chuẩn bị dữ liệu và thực thi thuật toán Federated Learning mới **FedProtoStd** (kết hợp **Prototype** và **Std** theo Định lý phương sai toàn phần & Gaussian Feature Augmentation).

---

## 1. Tổng Quan Thuật Toán FedProtoStd

Khác biệt cốt lõi so với **FedProto** truyền thống và **FedAvg**:
- **Không gửi trọng số mô hình**: Client giữ nguyên mô hình cá nhân hóa cục bộ, chỉ gửi lên Server bộ ba tham số phân bố cho từng nhãn: Prototype $\boldsymbol{\mu}_c \in \mathbb{R}^D$, Vector Std $\boldsymbol{\sigma}_c \in \mathbb{R}^D$, và số lượng mẫu $N_c$.
- **Tổng hợp Server theo Định lý phương sai toàn phần (Law of Total Variance)**:
  $$\boldsymbol{\sigma}_c^2 = \sum_k \frac{N_{k,c}}{N_c} \left[ \boldsymbol{\sigma}_{k,c}^2 + (\boldsymbol{\mu}_{k,c} - \boldsymbol{\mu}_c)^2 \right]$$
  Kết hợp cả độ phân tán nội bộ client (intra-client) và độ lệch đặc trưng giữa các client (inter-client drift).
- **Cơ chế huấn luyện kép tại Client (Dual Training)**:
  1. *Distribution-weighted Alignment Loss*: Căn chỉnh đặc trưng từ dữ liệu thật về Prototype toàn cục có trọng số nghịch đảo với phương sai (chiều đặc trưng nào ổn định thì ép chặt hơn).
  2. *Gaussian Feature Augmentation*: Lấy mẫu vector ảo $\tilde{\boldsymbol{z}} \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\sigma}_c^2)$ đưa qua Classifier Head để huấn luyện phân loại trên toàn bộ các lớp (khắc phục hoàn toàn lỗi missing classes khi dữ liệu Non-IID nặng).

---

## 2. Các File Mã Nguồn Thuật Toán

- **Client**: `system/flcore/clients/clientprotostd.py` (Lớp `clientProtoStd`)
- **Server**: `system/flcore/servers/serverprotostd.py` (Lớp `FedProtoStd`)
- **Main Driver**: `system/main.py` (Tích hợp `-algo FedProtoStd`)

---

## 3. Bảng Tham Số Cấu Hình

### 3.1. Các tham số đặc thù của FedProtoStd
| Tham số | Kiểu | Mặc định | Ý nghĩa & Khuyến nghị |
| :--- | :---: | :---: | :--- |
| `--lamda_align` | `float` | `1.0` | Hệ số phạt căn chỉnh đặc trưng cục bộ về Prototype toàn cục. Khuyến nghị: `0.5` - `1.0`. |
| `--lamda_aug` | `float` | `1.0` | Hệ số học bù từ đặc trưng ảo Gauss cho Classifier Head. Khuyến nghị: `0.5` - `2.0`. |
| `--num_aug_samples` | `int` | `10` | Số lượng mẫu ảo Gauss sinh ra cho mỗi class trong mỗi batch. Khuyến nghị: `5` - `20`. |
| `--eval_mode` | `str` | `head` | Cơ chế suy luận/đánh giá: `'head'` (dùng Classifier Head, khuyến nghị) hoặc `'proto'` (khoảng cách Mahalanobis NLL). |

### 3.2. Các tham số cơ bản của FLlib
| Tham số | Cờ viết tắt | Ý nghĩa | Ví dụ |
| :--- | :---: | :--- | :--- |
| `--algorithm` | `-algo` | Tên thuật toán | `FedProtoStd` hoặc `FedProto` |
| `--dataset` | `-data` | Bộ dữ liệu | `mnist`, `Cifar10`, `emnist` |
| `--model` | `-m` | Backbone mạng | `cnn`, `resnet18` |
| `--num_clients` | `-nc` | Tổng số client | `10` hoặc `20` |
| `--global_rounds` | `-gr` | Số vòng huấn luyện toàn cục | `50` hoặc `100` |
| `--local_epochs` | `-ls` | Số epoch huấn luyện cục bộ mỗi vòng | `1` |
| `--batch_size` | `-lbs` | Kích thước batch cục bộ | `32` hoặc `64` |
| `--local_learning_rate` | `-lr` | Tốc độ học (Learning rate) | `0.005` |
| `--device_id` | `-did` | GPU ID (`0`, `1`,... hoặc CPU nếu không có CUDA) | `0` |
| `--log` | `-log` | Bật logging trực tuyến lên WandB | Thêm cờ `-log` |

---

## 4. Chuẩn Bị Dữ Liệu (Dataset Generation)

Trước khi chạy thử nghiệm, bạn cần tạo dữ liệu phân mảnh cho các client. Di chuyển vào thư mục `dataset/`:

```bash
cd dataset
```

### Tạo MNIST Non-IID (Phân phối Dirichlet $\alpha=0.1$, 10 clients):
```bash
python3 generate_mnist.py noniid balance - 10 0.1
```

### Tạo MNIST IID (10 clients):
```bash
python3 generate_mnist.py iid balance - 10 0.1
```

### Tạo CIFAR-10 Non-IID (10 clients):
```bash
python3 generate_cifar10.py noniid balance - 10 0.1
```

---

## 5. Các Kịch Bản Thực Thi (Execution Scenarios)

Di chuyển vào thư mục `system/`:
```bash
cd system
```

### Kịch bản 1: Chạy thử nhanh kiểm tra hệ thống (2 clients, 5 rounds)
```bash
python3 main.py -data mnist -m cnn -algo FedProtoStd -nc 2 -gr 5 -ls 1 -lbs 64 -did 0
```

### Kịch bản 2: Huấn luyện chuẩn Non-IID (10 clients, 50 rounds, Classifier Head)
```bash
python3 main.py -data mnist -m cnn -algo FedProtoStd -nc 10 -gr 50 -ls 1 -lbs 32 -lr 0.005 -did 0 --eval_mode head --lamda_align 1.0 --lamda_aug 1.0 --num_aug_samples 10
```

### Kịch bản 3: Huấn luyện có ghi log trực tuyến lên WandB (`-log`)
```bash
python3 main.py -data mnist -m cnn -algo FedProtoStd -nc 10 -gr 50 -ls 1 -lbs 32 -did 0 --eval_mode head --lamda_align 1.0 --lamda_aug 1.0 -log
```
*Ghi chú: Toàn bộ biểu đồ Train Loss, Test Accuracy, Test AUC qua từng vòng sẽ được tự động đồng bộ lên project **`PFLA`** trên WandB.*

### Kịch bản 4: Đánh giá bằng khoảng cách Metric Prototype (`--eval_mode proto`)
```bash
python3 main.py -data mnist -m cnn -algo FedProtoStd -nc 10 -gr 50 -ls 1 -lbs 32 -did 0 --eval_mode proto --lamda_align 1.0 --lamda_aug 1.0
```

### Kịch bản 5: Chạy đối chuẩn so sánh với FedProto gốc
Để so sánh hiệu năng vượt trội của **FedProtoStd** so với **FedProto** trong cùng điều kiện:

```bash
# 1. Chạy FedProto gốc
python3 main.py -data mnist -m cnn -algo FedProto -nc 10 -gr 50 -ls 1 -lbs 32 -did 0 -log

# 2. Chạy FedProtoStd mới
python3 main.py -data mnist -m cnn -algo FedProtoStd -nc 10 -gr 50 -ls 1 -lbs 32 -did 0 --lamda_align 1.0 --lamda_aug 1.0 -log
```

---

## 6. Xem Kết Quả & Phân Tích

1. **File kết quả trên đĩa**:
   Sau khi hoàn tất, hệ thống tự động lưu file kết quả định dạng HDF5 vào thư mục `results/`:
   `results/mnist_FedProtoStd_test_0.h5`
   
   File chứa các mảng numpy:
   - `rs_test_acc`: Độ chính xác kiểm thử qua từng vòng.
   - `rs_test_auc`: Chỉ số ROC AUC qua từng vòng.
   - `rs_train_loss`: Hàm mất mát qua từng vòng.

2. **Đọc và vẽ đồ thị bằng Python**:
   ```python
   import h5py
   import matplotlib.pyplot as plt

   with h5py.File('../results/mnist_FedProtoStd_test_0.h5', 'r') as hf:
       acc = hf['rs_test_acc'][:]
       loss = hf['rs_train_loss'][:]

   plt.figure(figsize=(10, 4))
   plt.subplot(1, 2, 1)
   plt.plot(acc, label='Test Accuracy')
   plt.xlabel('Global Round')
   plt.ylabel('Accuracy')
   plt.grid(True)
   plt.legend()

   plt.subplot(1, 2, 2)
   plt.plot(loss, label='Train Loss', color='red')
   plt.xlabel('Global Round')
   plt.ylabel('Loss')
   plt.grid(True)
   plt.legend()
   plt.show()
   ```

3. **Xem trực tiếp trên WandB**:
   Truy cập: [WandB Project PFLA](https://wandb.ai/gvquang25-hanoi-university-of-science-and-technology/PFLA) để xem so sánh trực quan đa trục giữa các lần chạy.

---

## 7. Mẹo Tối Ưu Hiệu Năng (Tuning Tips)

1. **Khi dữ liệu Non-IID cực độ (mỗi client chỉ có 1-2 nhãn)**:
   - Tăng `--lamda_aug` lên `1.5` hoặc `2.0` để Classifier Head được học bù nhiều hơn từ đặc trưng ảo Gauss.
   - Tăng `--num_aug_samples` từ `10` lên `20`.
2. **Khi mô hình có xu hướng hội tụ chậm**:
   - Giảm nhẹ `--lamda_align` xuống `0.5` để mô hình ưu tiên học đặc trưng cục bộ trước khi bị ép chặt vào Prototype toàn cục.
3. **Tiết kiệm bộ nhớ GPU / Tăng tốc độ**:
   - Đặt `--num_aug_samples` bằng `5` hoặc `10`.

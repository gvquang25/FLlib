import h5py
import numpy as np
import os


def average_data(algorithm="", dataset="", goal="", times=10):
    test_acc = get_all_results_for_one_algo(algorithm, dataset, goal, times)

    max_accurancy = []
    for i in range(times):
        max_accurancy.append(test_acc[i].max())

    print("std for best accurancy:", np.std(max_accurancy))
    print("mean for best accurancy:", np.mean(max_accurancy))


def get_all_results_for_one_algo(algorithm="", dataset="", goal="", times=10):
    test_acc = []
    algorithms_list = [algorithm] * times
    for i in range(times):
        file_name = dataset + "_" + algorithms_list[i] + "_" + goal + "_" + str(i)
        test_acc.append(np.array(read_data_then_delete(file_name, delete=False)))

    return test_acc


def read_data_then_delete(file_name, delete=False):
    file_path = "../results/" + file_name + ".h5"

    with h5py.File(file_path, 'r') as hf:
        rs_test_acc = np.array(hf.get('rs_test_acc'))

    if delete:
        os.remove(file_path)
    print("Length: ", len(rs_test_acc))

    return rs_test_acc


def print_experiment_config(args, times_idx=None):
    """
    In bảng tóm tắt cấu hình thực nghiệm và cài đặt Non-IID ở đầu mỗi lượt chạy (run_i).
    Hỗ trợ ghim cấu hình lên WandB Dashboard nếu cờ -log được kích hoạt.
    """
    run_key = f"{getattr(args, 'algorithm', '')}_{times_idx}"
    if getattr(args, '_last_printed_run_key', None) == run_key:
        return
    args._last_printed_run_key = run_key

    width = 78
    sep_heavy = "=" * width
    sep_light = "-" * width

    algo = getattr(args, 'algorithm', 'Unknown')
    dataset = getattr(args, 'dataset', 'Unknown')
    num_classes = getattr(args, 'num_classes', 'N/A')
    model = getattr(args, 'model_str', getattr(args, 'model', 'Unknown'))
    device = str(getattr(args, 'device', 'cpu'))
    times = getattr(args, 'times', 1)
    run_str = f"Run {times_idx + 1}/{times} (Time index: {times_idx})" if times_idx is not None else f"Total runs: {times}"
    seed_val = getattr(args, 'current_seed', getattr(args, 'seed', 'N/A'))
    shuffle_flag = getattr(args, 'shuffle', False)

    noniid = getattr(args, 'noniid', False)
    dist_str = "Non-IID (Dirichlet partition)" if noniid else "IID (Uniform partition)"
    alpha = getattr(args, 'alpha_dirich', 'N/A')
    balance = getattr(args, 'balance', False)
    num_clients = getattr(args, 'num_clients', 'N/A')
    join_ratio = getattr(args, 'join_ratio', 1.0)
    join_cnt = int(num_clients * join_ratio) if isinstance(num_clients, int) else 'N/A'
    random_join = getattr(args, 'random_join_ratio', False)

    gr = getattr(args, 'global_rounds', 'N/A')
    ls = getattr(args, 'local_epochs', 'N/A')
    bs = getattr(args, 'batch_size', 'N/A')
    lr = getattr(args, 'local_learning_rate', 'N/A')
    lr_decay = getattr(args, 'learning_rate_decay', False)
    gamma = getattr(args, 'learning_rate_decay_gamma', 0.99)
    lr_str = f"{lr} (Decay ExponentialLR: gamma={gamma})" if lr_decay else f"{lr} (Fixed)"

    enc_lr = getattr(args, 'encoder_lr', None)
    enc_lr_str = f"{enc_lr} (Separate Optimizer)" if (enc_lr is not None and enc_lr > 0) else f"Inherited lr={lr} (Separate Optimizer)"
    lam = getattr(args, 'lamda', 'N/A')
    lam_proto = getattr(args, 'lamda_proto', 'N/A')
    nct = getattr(args, 'infoNCET', 'N/A')
    warmup = getattr(args, 'proto_warmup', 'N/A')
    eg = getattr(args, 'eval_gap', 1)
    log_flag = getattr(args, 'log', False)

    lines = [
        sep_heavy,
        f"{'FEDERATED LEARNING EXPERIMENT SUMMARY':^{width}}",
        sep_heavy,
        " [1]  CONFIGURATION",
        f"  * Algorithm         : {algo}",
        f"  * Dataset          : {dataset} ({num_classes} classes)",
        f"  * Backbone      : {model}",
        f"  * Device    : {device}",
        f"  * Current Run       : {run_str}",
        f"  * Random Seed       : {seed_val}",
        f"  * Shuffle Train     : {'Enabled' if shuffle_flag else 'Disabled'}",
        sep_light,
        " [2] NON-IID SETTINGS",
        f"  * Distribution: {dist_str}",
        f"  * Dirichlet Alpha  : {alpha}",
        f"  * Balance     : {balance}",
        f"  * Number Client (-nc)           : {num_clients}",
        f"  * join_ratio    : {join_ratio} (~{join_cnt} clients/round)",
        f"  * Random Join Ratio              : {random_join}",
        sep_light,
        " [3] HYPERPARAMETERS",
        f"  * Global Rounds (-gr)           : {gr}",
        f"  * Local Epochs (-ls)            : {ls}",
        f"  * Local Batch Size (-lbs)        : {bs}",
        f"  * Head Learning Rate (-lr)       : {lr_str}",
        f"  * Encoder Learning Rate          : {enc_lr_str}",
        f"  * Prototype Loss Weight (-lp)    : {lam_proto} (InfoNCE + MSE)",
        f"  * InfoNCE Temperature (-nct)    : {nct}",
        f"  * Proto Warmup Rounds            : {warmup} round(s)",
        f"  * Regularization Weight (-lam)   : {lam}",
        f"  * Evaluation Frequency (-eg)     : Every {eg} round(s)",
        f"  * WandB Logging (-log)           : {'Enabled' if log_flag else 'Disabled'}",
        sep_heavy
    ]
    print("\n" + "\n".join(lines) + "\n")

    # Ghim cau hinh vao WandB Config neu dang su dung
    if log_flag:
        try:
            import wandb
            if not hasattr(wandb, 'init'):
                wandb = None
            if wandb is not None and wandb.run is not None:
                wandb.config.update({
                    "summary/algorithm": algo,
                    "summary/dataset": dataset,
                    "summary/model": str(model),
                    "summary/run_index": times_idx,
                    "summary/seed": seed_val,
                    "summary/shuffle": shuffle_flag,
                    "summary/distribution": dist_str,
                    "summary/alpha_dirich": alpha,
                    "summary/balance": balance,
                    "summary/num_clients": num_clients,
                    "summary/join_ratio": join_ratio,
                    "summary/global_rounds": gr,
                    "summary/local_epochs": ls,
                    "summary/batch_size": bs,
                    "summary/head_lr": lr,
                    "summary/encoder_lr": enc_lr if enc_lr is not None else lr,
                    "summary/lamda_proto": lam_proto,
                    "summary/infoNCET": nct,
                }, allow_val_change=True)
        except Exception:
            pass

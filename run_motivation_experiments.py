import subprocess
import sys
import os
import time

EXPERIMENTS = [
    {
        "name": "FedSCD (Ours)",
        "cmd": [
            sys.executable, "main.py",
            "-data", "Cifar10", "-m", "cnn", "-algo", "FedSCD",
            "-gr", "100", "-did", "0", "-lbs", "32", "-tbs", "512",
            "-lr", "0.01", "-nc", "20", "--noniid", "--alpha_dirich", "0.5",
            "-lam", "0.1", "-sd", "42", "-log",
            "--proj_mode", "shared", "--lamda_virt_local", "0.0"
        ]
    },
    {
        "name": "FedProto",
        "cmd": [
            sys.executable, "main.py",
            "-data", "Cifar10", "-m", "cnn", "-algo", "FedProto",
            "-gr", "100", "-did", "0", "-lbs", "32", "-tbs", "512",
            "-lr", "0.01", "-nc", "20", "--noniid", "--alpha_dirich", "0.5",
            "-lam", "0.1", "-sd", "42", "-log"
        ]
    },
    {
        "name": "FedTGP",
        "cmd": [
            sys.executable, "main.py",
            "-data", "Cifar10", "-m", "cnn", "-algo", "FedTGP",
            "-gr", "100", "-did", "0", "-lbs", "32", "-tbs", "512",
            "-lr", "0.01", "-nc", "20", "--noniid", "--alpha_dirich", "0.5",
            "-lam", "0.1", "-sd", "42", "-log"
        ]
    },
    {
        "name": "CCVR",
        "cmd": [
            sys.executable, "main.py",
            "-data", "Cifar10", "-m", "cnn", "-algo", "CCVR",
            "-gr", "100", "-did", "0", "-lbs", "32", "-tbs", "512",
            "-lr", "0.01", "-nc", "20", "--noniid", "--alpha_dirich", "0.5",
            "-lam", "0.1", "-sd", "42", "-log"
        ]
    }
]


def run_all(global_rounds=None):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    system_dir = os.path.join(base_dir, "system")

    total_start = time.time()
    for idx, exp in enumerate(EXPERIMENTS, start=1):
        cmd = list(exp["cmd"])
        if global_rounds is not None:
            # Override global rounds if specified
            if "-gr" in cmd:
                gr_idx = cmd.index("-gr")
                cmd[gr_idx + 1] = str(global_rounds)

        print("=" * 70)
        print(f"[{idx}/{len(EXPERIMENTS)}] RUNNING: {exp['name']}")
        print("Command:", " ".join(cmd))
        print("=" * 70)

        start = time.time()
        ret = subprocess.run(cmd, cwd=system_dir)
        elapsed = time.time() - start
        if ret.returncode != 0:
            print(f"[ERROR] {exp['name']} exited with code {ret.returncode}")
            sys.exit(ret.returncode)
        print(f"[COMPLETED] {exp['name']} in {elapsed:.1f}s\n")

    print("=" * 70)
    print("All experiments completed! Generating Figure 1...")
    print("=" * 70)

    plot_cmd = [sys.executable, os.path.join(base_dir, "plot_motivation.py")]
    subprocess.run(plot_cmd, cwd=base_dir)
    print(f"\n[DONE] Pipeline finished in {(time.time() - total_start) / 60:.1f} minutes.")


if __name__ == "__main__":
    gr = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_all(global_rounds=gr)

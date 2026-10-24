import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

ALGO_CONFIG = [
    ("FedProto", "FedProto", "#7f7f7f", "--", 1.6),
    ("CCVR", "CCVR", "#2ca02c", ":", 1.7),
    ("FedTGP", "FedTGP", "#d95f02", "-.", 1.8),
    ("FedNewPCL", "FedSCD (Ours)", "#0072B2", "-", 2.3),
]


def ema_smooth(values, weight=0.6):
    if len(values) == 0:
        return values
    last = values[0]
    smoothed = []
    for v in values:
        s = last * weight + (1.0 - weight) * v
        smoothed.append(s)
        last = s
    return np.array(smoothed)


def load_histories(results_dir="results", dataset="Cifar10", goal="test", run_idx=0):
    histories = {}
    for algo_code, display_name, _, _, _ in ALGO_CONFIG:
        h5_path = os.path.join(results_dir, f"{dataset}_{algo_code}_{goal}_{run_idx}.h5")
        if os.path.exists(h5_path):
            with h5py.File(h5_path, "r") as hf:
                if "rs_proto_margin" not in hf or "rs_bhat_overlap" not in hf:
                    print(f"[SKIP] {h5_path} does not contain rs_proto_margin or rs_bhat_overlap.")
                    continue
                acc = np.array(hf["rs_test_acc"])
                margin = np.array(hf["rs_proto_margin"])
                overlap = np.array(hf["rs_bhat_overlap"])
                histories[display_name] = {
                    "rounds": np.arange(len(acc)),
                    "margin": margin,
                    "overlap": overlap,
                    "acc": acc * 100.0 if acc.max() <= 1.0 else acc,
                }
                print(f"[OK - H5] Loaded {h5_path} -> {display_name} ({len(acc)} rounds)")
    return histories


def plot_icml_figure(histories, out_pdf="figures/motivation.pdf"):
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "axes.titlesize": 9.5,
        "legend.fontsize": 9,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.linewidth": 0.8,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig, axes = plt.subplots(1, 3, figsize=(6.75, 2.05), sharex=True)

    panels = [
        ("margin", "Prototype Margin", "Left: Prototype Margin"),
        ("overlap", "Bhattacharyya Overlap", "Middle: Bhattacharyya Overlap"),
        ("acc", "Test Accuracy (%)", "Right: Test Accuracy"),
    ]

    for ax, (key, ylabel, title) in zip(axes, panels):
        for _, display_name, color, lstyle, lwidth in ALGO_CONFIG:
            if display_name not in histories:
                continue
            r = histories[display_name]["rounds"]
            y_raw = histories[display_name][key]
            y_smooth = ema_smooth(y_raw, weight=0.6)

            ax.plot(r, y_raw, color=color, alpha=0.2, linewidth=0.8)
            ax.plot(r, y_smooth, label=display_name, color=color, linestyle=lstyle, linewidth=lwidth)

        ax.set_xlabel("Communication Rounds")
        ax.set_ylabel(ylabel)
        ax.set_title(title, pad=6)
        ax.grid(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.13), ncol=4, frameon=False)

    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    png_path = out_pdf.replace(".pdf", ".png")
    plt.savefig(png_path, format="png", dpi=300, bbox_inches="tight")
    print(f"\n[DONE] Saved Figure 1 to: {os.path.abspath(out_pdf)}")
    print(f"[DONE] Saved preview image to: {os.path.abspath(png_path)}")

    # Upload len WandB duoi dang Image va Artifact
    try:
        import wandb
        print("[WandB] Uploading Figure 1 to WandB project 'PFLA'...")
        run = wandb.init(
            project="PFLA",
            name="Figure_1_Motivation",
            job_type="visualization",
            reinit=True
        )
        wandb.log({
            "figures/motivation": wandb.Image(png_path, caption="Figure 1: Motivation (Prototype Margin, Bhattacharyya Overlap, Test Accuracy)")
        })
        artifact = wandb.Artifact(name="figure1_motivation", type="figure")
        artifact.add_file(out_pdf)
        artifact.add_file(png_path)
        run.log_artifact(artifact)
        wandb.finish()
        print("[WandB] Successfully uploaded Figure 1 (Image & Artifact) to WandB!")
    except Exception as e:
        print(f"[WandB Warning] Could not upload Figure 1 to WandB: {e}")


if __name__ == "__main__":
    hist = load_histories(results_dir="results")
    if hist:
        plot_icml_figure(hist, out_pdf="figures/motivation.pdf")
    else:
        print("No valid .h5 files found in results/. Run Step 4 first.")

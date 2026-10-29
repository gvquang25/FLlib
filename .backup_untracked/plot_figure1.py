"""
plot_figure1.py -- ve Figure 1 (2 hang x 3 cot) tu cac file overlap_*.json.
Chay:  python plot_figure1.py
Sua dict CONFIG ben duoi cho dung ten file/thuat toan cua ban.
"""
import glob, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULT_DIR = "../results"
# hang -> tag dataset (phai trung 'tag' khi tao OverlapLogger)
ROWS = [("Label skew (CIFAR-10, Dir 0.5)", "Cifar10"),
        ("Domain skew (Digit-5)", "Digit5")]
# ten thuat toan trong repo -> (nhan trong hinh, mau, kieu net)
METHODS = {"FedProto": ("FedProto", "#4C72B0", "--"),
           "FedTGP": ("FedTGP", "#8172B2", ":"),
           "FedCCVR": ("CCVR", "#CCB974", "--"),
           "FedNewProto": ("FedNewProto", "#55A868", "-."),
           "FedNewPCL": ("FedSCD (ours)", "#C44E52", "-")}
COLS = [("margin_rel", "Prototype margin (relative)  $\\uparrow$"),
        ("bhat_mean_per_dim", "Bhattacharyya distance / dim  $\\uparrow$"),
        ("test_acc", "Test accuracy  $\\uparrow$")]
SMOOTH = 3          # TB truot de duong cong muot (1 = tat)

plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "legend.fontsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.3})

def load(tag, algo):
    runs = []
    for p in sorted(glob.glob(f"{RESULT_DIR}/overlap_{tag}_{algo}_seed*.json")):
        runs.append(json.load(open(p)))
    return runs

def smooth(y, k):
    if k <= 1: return y
    pad = np.pad(y, (k // 2, k - 1 - k // 2), mode="edge")
    return np.convolve(pad, np.ones(k) / k, mode="valid")

fig, axes = plt.subplots(len(ROWS), 3, figsize=(6.75, 1.75 * len(ROWS)), squeeze=False)
for r, (rname, tag) in enumerate(ROWS):
    for c, (key, ylabel) in enumerate(COLS):
        ax = axes[r, c]
        for algo, (lab, col, ls) in METHODS.items():
            runs = load(tag, algo)
            if not runs: continue
            L = min(len(x[key]) for x in runs)
            Y = np.array([[np.nan if v is None else v for v in x[key][:L]] for x in runs], float)
            x = np.array(runs[0]["round"][:L])
            m = smooth(np.nanmean(Y, 0), SMOOTH)
            ax.plot(x, m, ls, color=col, label=lab)
            if len(runs) > 1:
                s = smooth(np.nanstd(Y, 0), SMOOTH)
                ax.fill_between(x, m - s, m + s, color=col, alpha=0.15, lw=0)
        ax.grid(alpha=0.25, lw=0.4)
        if r == len(ROWS) - 1: ax.set_xlabel("Communication round")
        ax.set_ylabel(ylabel if c else f"{rname}\n{ylabel}")
        if r == 0 and c == 0: ax.legend(frameon=False, loc="best")
fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
import os; os.makedirs("figures", exist_ok=True)
fig.savefig("figures/motivation.pdf", bbox_inches="tight")
fig.savefig("figures/motivation.png", dpi=300, bbox_inches="tight")
print("saved figures/motivation.pdf")

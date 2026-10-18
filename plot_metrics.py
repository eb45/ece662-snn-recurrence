"""
Plot training curves from one or more runs' metrics.jsonl files.

One run:
    python plot_metrics.py /work/eb408/ece662_runs/control_s2026

Compare runs (test accuracy solid, train accuracy dashed, same color per run):
    python plot_metrics.py runA runB runC --labels control lsm recurrent --out comparison.png

The default output is <first run dir>/curves.png.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")  # no display needed on the cluster
import matplotlib.pyplot as plt


def load_metrics(run_dir):
    path = os.path.join(run_dir, "metrics.jsonl")
    rows = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                rows[r["epoch"]] = r  # if an epoch was logged twice, keep the last
    return [rows[k] for k in sorted(rows)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="Run folders containing metrics.jsonl")
    p.add_argument("--labels", nargs="+", help="One label per run (default: folder names)")
    p.add_argument("--out", help="Output PNG path")
    args = p.parse_args()

    labels = args.labels or [os.path.basename(os.path.normpath(r)) for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels needs one label per run")
    out = args.out or os.path.join(args.runs[0], "curves.png")

    fig, (ax_acc, ax_loss) = plt.subplots(1, 2, figsize=(11, 4))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (run, label) in enumerate(zip(args.runs, labels)):
        rows = load_metrics(run)
        ep = [r["epoch"] + 1 for r in rows]  # 1-indexed for display
        c = colors[i % len(colors)]
        test = [r["test_acc"] for r in rows]
        ax_acc.plot(ep, test, color=c, label=f"{label} test")
        ax_acc.plot(ep, [r["train_acc"] for r in rows], color=c, linestyle="--", alpha=0.6,
                    label=f"{label} train")
        ax_loss.plot(ep, [r["train_loss"] for r in rows], color=c, label=label)
        print(f"{label}: best test {max(test):.4f}, final test {test[-1]:.4f}, "
              f"epochs logged {len(rows)}")

    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("top-1 accuracy")
    ax_acc.set_title("Accuracy")
    ax_acc.legend(fontsize=8)
    ax_acc.grid(alpha=0.3)

    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("train loss")
    ax_loss.set_yscale("log")
    ax_loss.set_title("Training loss")
    ax_loss.legend(fontsize=8)
    ax_loss.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()

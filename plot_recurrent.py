"""
Two-panel figure for the recurrent-SNN section of the report.

(a) test accuracy per epoch: control, frozen random recurrence, fc learned at
    the network's lr, fc learned with W_rec lr 1e-4
(b) norm of the fc W_rec per epoch for the same recurrent runs plus weight decay

Sized for one column of the ACM two-column template (3.33 in wide).

    python plot_recurrent.py --runs-dir /work/ah670/ece662_runs \
        --control /path/to/control_s2026 --out figs/recurrent_curves.png

--control is the folder holding the control's metrics.jsonl (Emma's run). If it
is left out, panel (a) is drawn without the control. Use a .pdf --out for a
vector figure.
"""

import argparse
import os

import matplotlib.pyplot as plt

from plot_metrics import load_metrics

# Color follows the run, in the same slot order on both panels.
SERIES = [
    # (label, run folder name or None for the control, color)
    ("Control (no recurrence)", None, "#2a78d6"),
    ("Frozen random $W_{\\mathrm{rec}}$", "rec_conv5_fc_frozen_s2026", "#eb6834"),
    ("Learned, same lr", "rec_fc_s2026", "#1baf7a"),
    ("Learned, $W_{\\mathrm{rec}}$ lr $10^{-4}$", "rec_fc_rlr1e-4_s2026", "#eda100"),
    ("Learned, weight decay 0.1", "rec_fc_wd0.1_s2026", "#e87ba4"),
]
IN_PANEL_A = {"Control (no recurrence)", "Frozen random $W_{\\mathrm{rec}}$",
              "Learned, same lr", "Learned, $W_{\\mathrm{rec}}$ lr $10^{-4}$"}

INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def style_axes(ax):
    ax.grid(True, axis="y", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, length=2, width=0.6)
    ax.set_xlim(1, 64)
    ax.set_xticks([1, 16, 32, 48, 64])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/work/ah670/ece662_runs")
    p.add_argument("--control", help="Folder with the control's metrics.jsonl")
    p.add_argument("--out", default="figs/recurrent_curves.png")
    args = p.parse_args()

    plt.rcParams.update({"font.size": 7, "axes.labelcolor": INK, "text.color": INK,
                         "axes.titlesize": 7.5, "legend.fontsize": 6.5})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.33, 4.1), sharex=True,
                                     constrained_layout=True)

    handles = []
    for label, run, color in SERIES:
        if run is None:
            if not args.control or not os.path.exists(os.path.join(args.control, "metrics.jsonl")):
                print("no control metrics.jsonl found; panel (a) drawn without the control")
                continue
            rows = load_metrics(args.control)
        else:
            path = os.path.join(args.runs_dir, run)
            if not os.path.exists(os.path.join(path, "metrics.jsonl")):
                print(f"skipping {run}: no metrics.jsonl")
                continue
            rows = load_metrics(path)
        ep = [r["epoch"] + 1 for r in rows]

        line = None
        if label in IN_PANEL_A:
            (line,) = ax_a.plot(ep, [100 * r["test_acc"] for r in rows], color=color,
                                linewidth=1.1, solid_joinstyle="round", solid_capstyle="round")
        if run is not None:
            norms = [r["rec_weight_norm"]["fc"] for r in rows]
            (line_b,) = ax_b.plot(ep, norms, color=color, linewidth=1.1,
                                  solid_joinstyle="round", solid_capstyle="round")
            line = line or line_b
            print(f"{label}: test best {100 * max(r['test_acc'] for r in rows):.1f}, "
                  f"final {100 * rows[-1]['test_acc']:.1f}; fc |W_rec| {norms[0]:.2f} -> {norms[-1]:.2f}")
        handles.append((line, label))

    style_axes(ax_a)
    ax_a.set_ylim(45, 100)
    ax_a.set_ylabel("Test accuracy (%)")
    ax_a.set_title("(a) Test accuracy on unseen people", loc="left")

    style_axes(ax_b)
    ax_b.set_ylim(bottom=0)
    ax_b.set_xlabel("Epoch")
    ax_b.set_ylabel(r"$\|W_{\mathrm{rec}}\|$ (fc layer)")
    ax_b.set_title(r"(b) Growth of the recurrent weights", loc="left")

    fig.legend([h for h, _ in handles], [l for _, l in handles], loc="outside upper center",
               ncol=2, frameon=False, handlelength=1.6, columnspacing=1.2)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=300, facecolor="white")
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()

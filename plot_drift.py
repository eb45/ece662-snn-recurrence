"""
Does how far W_rec moves from its init predict how badly it generalizes?

Collects every recurrent run under --runs-dir and writes
  - <out>.csv : one row per run (config, final train/test acc, drift, spectral norm)
  - a printed table of final-epoch test accuracy mean +/- std over seeds per config
  - <out>.png : (a) final test accuracy vs relative drift ||W - W0|| / ||W0||
                (b) final test accuracy vs spectral norm of W_rec (max over layers)
    colour = placement, marker = condition (lr grid, decay toward 0, anchored, frozen)
  - <lr-out>.png : final test accuracy vs W_rec learning rate, one line per placement,
                   mean +/- std over seeds, frozen as a separate point (DVS128, 64-epoch runs)

Drift comes from the last row of metrics.jsonl (runs trained with drift logging)
or from drift.json (older runs; make it with `python rec_drift.py <runs>`).

    python plot_drift.py --runs-dir /work/ah670/ece662_runs \
        --control /path/to/control_s2026 --out figs/drift_vs_acc.png

The plot keeps the main setting only (mean readout, orthogonal init, gain 0.5,
no grad clipping); everything is still listed in the CSV.
"""

import argparse
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from plot_metrics import load_metrics

# Same palette and order as plot_recurrent.py; colour follows the placement.
PLACEMENT_COLORS = {"fc": "#2a78d6", "conv5": "#eb6834", "conv5+fc": "#1baf7a"}
CONDITION_MARKERS = {"lr grid": "o", "decay to 0": "s", "anchored": "^", "frozen": "D"}
INK, INK_2, GRID, AXIS = "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"


def condition(a: dict) -> str:
    if a["freeze_rec"]:
        return "frozen"
    if (a.get("rec_wd") or 0) > 0:
        return "anchored" if a.get("rec_anchor") else "decay to 0"
    return "lr grid"


def collect(runs_dir: str) -> list:
    rows = []
    for final_path in sorted(glob.glob(os.path.join(runs_dir, "*", "final.json"))):
        run_dir = os.path.dirname(final_path)
        with open(final_path) as f:
            final = json.load(f)
        a = final["args"]
        if "rec_layers" not in a:
            continue  # control or LSM
        metrics = load_metrics(run_dir)
        last = metrics[-1] if metrics else {}

        stats, source = last.get("rec_drift"), "metrics"
        if stats is None and os.path.exists(os.path.join(run_dir, "drift.json")):
            with open(os.path.join(run_dir, "drift.json")) as f:
                stats, source = json.load(f).get("final"), "drift.json"
        if stats is None:
            source = "missing"

        main_setting = (a["readout"] == "mean" and a["rec_init"] == "orthogonal"
                        and a["rec_gain"] == 0.5 and not a.get("clip_grad"))
        rows.append({
            "run": os.path.basename(run_dir),
            "placement": "+".join(a["rec_layers"]),
            "condition": condition(a),
            "seed": a["seed"],
            "dataset": a.get("dataset", "dvs128"),
            "epochs_planned": a.get("epochs"),
            "epochs_done": len(metrics),
            "rec_lr": a.get("rec_lr") or a["lr"],  # old runs predate --rec-lr
            "rec_wd": a.get("rec_wd") or 0.0,
            "readout": a["readout"],
            "init": f"{a['rec_init']}x{a['rec_gain']}",
            "main_setting": main_setting,
            "train_acc": last.get("train_acc"),
            "best_test_acc": final["best_test_acc"],
            "final_test_acc": final["final_test_acc"],
            "drift_rel": stats["all"]["drift_rel"] if stats else None,
            "drift_abs": stats["all"]["drift_abs"] if stats else None,
            "spec_norm_max": stats["all"]["spec_norm_max"] if stats else None,
            "fc_radius": stats.get("fc", {}).get("radius") if stats else None,
            "fc_stable_rank": stats.get("fc", {}).get("stable_rank") if stats else None,
            "drift_source": source,
        })
    return rows


def print_table(rows: list) -> None:
    groups = defaultdict(list)
    for r in rows:
        key = (r["placement"], r["condition"], r["rec_lr"], r["rec_wd"], r["readout"], r["init"])
        groups[key].append(r)
    print(f"\n{'placement':9} {'condition':10} {'rec lr':>7} {'wd':>4} {'readout':7} "
          f"{'n':>2} {'final test %':>13} {'train-test':>10} {'drift':>6}")
    for key in sorted(groups, key=lambda k: (k[0], k[1], -k[2], k[3])):
        g = groups[key]
        accs = [100 * r["final_test_acc"] for r in g]
        gaps = [100 * (r["train_acc"] - r["final_test_acc"]) for r in g if r["train_acc"] is not None]
        drifts = [r["drift_rel"] for r in g if r["drift_rel"] is not None]
        sd = f"{statistics.stdev(accs):.1f}" if len(accs) > 1 else "-"
        print(f"{key[0]:9} {key[1]:10} {key[2]:7.0e} {key[3]:4g} {key[4]:7} {len(g):2d} "
              f"{statistics.mean(accs):7.1f} ± {sd:>3} "
              f"{statistics.mean(gaps) if gaps else float('nan'):10.1f} "
              f"{statistics.mean(drifts) if drifts else float('nan'):6.2f}")


def control_final_acc(control_dir):
    if not control_dir:
        return None
    rows = load_metrics(control_dir)
    return 100 * rows[-1]["test_acc"] if rows else None


def style_axes(ax):
    ax.grid(True, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, length=2, width=0.6)


def plot(rows: list, control_acc, out: str) -> None:
    pts = [r for r in rows if r["main_setting"] and r["spec_norm_max"] is not None]
    skipped = [r["run"] for r in rows if r["main_setting"] and r["spec_norm_max"] is None]
    if skipped:
        print(f"\nnot plotted (no drift stats; run rec_drift.py on them): {' '.join(skipped)}")

    plt.rcParams.update({"font.size": 7, "axes.labelcolor": INK, "text.color": INK,
                         "axes.titlesize": 7.5, "legend.fontsize": 6.5})
    fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(3.33, 4.6), sharey=True,
                                     constrained_layout=True)
    for ax, x_key in ((ax_a, "drift_rel"), (ax_b, "spec_norm_max")):
        for r in pts:
            if r[x_key] is None:
                continue
            ax.scatter(r[x_key], 100 * r["final_test_acc"], s=22,
                       marker=CONDITION_MARKERS[r["condition"]],
                       color=PLACEMENT_COLORS.get(r["placement"], INK_2),
                       edgecolors="white", linewidths=0.6, zorder=3)
        if control_acc is not None:
            ax.axhline(control_acc, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
            ax.annotate("control", xy=(1, control_acc), xycoords=("axes fraction", "data"),
                        xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                        color=INK_2, fontsize=6.5)
        style_axes(ax)
        ax.set_ylabel("Final test accuracy (%)")

    ax_a.set_xlabel(r"Relative drift $\|W_{\mathrm{rec}} - W_0\| \,/\, \|W_0\|$")
    ax_a.set_title("(a) Accuracy vs. how far $W_{\\mathrm{rec}}$ moved", loc="left")
    ax_b.set_xlabel(r"Spectral norm of $W_{\mathrm{rec}}$ (max over layers)")
    ax_b.set_title("(b) Accuracy vs. recurrent gain", loc="left")

    handles = [Line2D([], [], linestyle="", marker="o", markersize=5, color=c, label=p)
               for p, c in PLACEMENT_COLORS.items()]
    handles += [Line2D([], [], linestyle="", marker=m, markersize=5, color=INK_2, label=c)
                for c, m in CONDITION_MARKERS.items()]
    fig.legend(handles=handles, loc="outside upper center", ncol=4, frameon=False,
               handletextpad=0.3, columnspacing=1.0)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")


def lr_curves(rows: list, dataset: str = "dvs128", min_epochs: int = 64) -> dict:
    """Final test accuracy vs W_rec lr, one curve per placement.

    Keeps finished main-setting runs of one dataset that are either in the lr grid
    (no weight decay) or frozen. Returns
      {placement: {"lr": [(rec_lr, mean %, std %, n), ...] sorted by lr,
                   "frozen": (mean %, std %, n) or None}}
    with mean and std over seeds (std 0 for a single seed).
    """
    groups = defaultdict(list)
    for r in rows:
        if not (r["main_setting"] and r.get("dataset", "dvs128") == dataset
                and (r.get("epochs_done") or 0) >= min_epochs
                and r["condition"] in ("lr grid", "frozen")):
            continue
        key = "frozen" if r["condition"] == "frozen" else r["rec_lr"]
        groups[(r["placement"], key)].append(100 * r["final_test_acc"])

    def stats(accs):
        return (statistics.mean(accs), statistics.stdev(accs) if len(accs) > 1 else 0.0, len(accs))

    out = {}
    for (place, key), accs in groups.items():
        c = out.setdefault(place, {"lr": [], "frozen": None})
        if key == "frozen":
            c["frozen"] = stats(accs)
        else:
            c["lr"].append((key,) + stats(accs))
    for c in out.values():
        c["lr"].sort()
    return out


def plot_lr(rows: list, control_acc, out: str) -> None:
    """Accuracy vs W_rec learning rate (log x), one line per placement, mean +/- std over seeds.

    Frozen W_rec (lr 0) cannot sit on a log axis, so it is drawn as an unconnected
    point at a separate "frozen" tick left of the grid.
    """
    curves = lr_curves(rows)
    if not curves:
        print("\nno finished DVS128 lr-grid runs; skipping the lr plot")
        return
    lrs = sorted({p[0] for c in curves.values() for p in c["lr"]})
    x_frozen = (lrs[0] if lrs else 1e-4) / 3.5

    plt.rcParams.update({"font.size": 7, "axes.labelcolor": INK, "text.color": INK,
                         "axes.titlesize": 7.5, "legend.fontsize": 6.5})
    fig, ax = plt.subplots(figsize=(3.33, 2.4), constrained_layout=True)
    order = [p for p in PLACEMENT_COLORS if p in curves] + sorted(set(curves) - set(PLACEMENT_COLORS))
    for i, place in enumerate(order):
        c, color = curves[place], PLACEMENT_COLORS.get(place, INK_2)
        nudge = 1.06 ** (i - (len(order) - 1) / 2)  # keep overlapping error bars apart
        if c["lr"]:
            x = [p[0] * nudge for p in c["lr"]]
            ax.errorbar(x, [p[1] for p in c["lr"]], yerr=[p[2] for p in c["lr"]],
                        color=color, marker="o", markersize=4, linewidth=1.2, capsize=2,
                        elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.5,
                        label=place, zorder=3)
        if c["frozen"]:
            m, sd, _ = c["frozen"]
            ax.errorbar([x_frozen * nudge], [m], yerr=[sd], color=color,
                        marker=CONDITION_MARKERS["frozen"], markersize=4, linestyle="",
                        capsize=2, elinewidth=0.8, markeredgecolor="white",
                        markeredgewidth=0.5, label=None if c["lr"] else place, zorder=3)
    if control_acc is not None:
        ax.axhline(control_acc, color=INK_2, linewidth=0.8, linestyle=(0, (3, 2)), zorder=2)
        ax.annotate("control", xy=(1, control_acc), xycoords=("axes fraction", "data"),
                    xytext=(-2, 2), textcoords="offset points", ha="right", va="bottom",
                    color=INK_2, fontsize=6.5)

    ax.set_xscale("log")
    ticks = [x_frozen] + lrs
    ax.set_xticks(ticks)
    ax.set_xticklabels(["frozen"] + [f"{lr:g}" for lr in lrs])
    ax.minorticks_off()
    ax.set_xlim(x_frozen / 1.8, (lrs[-1] if lrs else x_frozen) * 1.8)
    style_axes(ax)
    ax.set_xlabel(r"$W_{\mathrm{rec}}$ learning rate (network lr $10^{-3}$)")
    ax.set_ylabel("Final test accuracy (%, mean ± std)")
    ax.set_title("Accuracy vs. $W_{\\mathrm{rec}}$ learning rate (DVS128)", loc="left")
    ax.legend(frameon=False, loc="lower left", handletextpad=0.4)

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=300, facecolor="white")
    print(f"Saved {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="/work/ah670/ece662_runs")
    p.add_argument("--control", help="Folder with the control's metrics.jsonl (dashed line)")
    p.add_argument("--out", default="figs/drift_vs_acc.png")
    p.add_argument("--lr-out", default="figs/acc_vs_rec_lr.png",
                   help="Accuracy vs W_rec learning rate, one line per placement")
    args = p.parse_args()

    rows = collect(args.runs_dir)
    if not rows:
        raise SystemExit(f"no recurrent runs with final.json under {args.runs_dir}")
    csv_path = os.path.splitext(args.out)[0] + ".csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_path} ({len(rows)} runs)")
    # The CSV lists everything; the table and plots skip debug and unfinished runs.
    full = [r for r in rows if r["epochs_done"] >= 64]
    print_table(full)
    control_acc = control_final_acc(args.control)
    plot(full, control_acc, args.out)
    plot_lr(rows, control_acc, args.lr_out)


if __name__ == "__main__":
    main()

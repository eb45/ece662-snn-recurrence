"""
Gap sweep figure: test accuracy vs number of empty frames inserted mid-clip.

One column per dataset (DVS128, DVS-GC 96-u), one line per arm A to E,
mean +/- std over seeds. Top row: test accuracy, with chance as a dashed line.
Bottom row: accuracy relative to the same run's gap-0 accuracy, so every arm
starts at 1 and the curves show how much each arm loses as the gap grows. Reads results/summary.csv
for the arm labels and results/runs/<run>/gap.json for the gap scores, so run
collect_results.py first.

    python plot_gap.py
    python plot_gap.py --datasets dvs128 dvsgc-96u dvsgc-81p --out figs/gap_sweep.png

Also writes <out>.csv with the per-arm means and stds that went into the plot.
"""

import argparse
import csv
import json
import os
import statistics
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

ARMS = {
    "A": "A: control",
    "B": "B: frozen W_rec",
    "C": "C: learned W_rec (lr 1e-3)",
    "D": "D: slow W_rec (lr 1e-4)",
    "E": "E: anchored W_rec (wd 1)",
}
TITLES = {"dvs128": "DVS128 Gesture (T=16)", "dvsgc-96u": "DVS-GC 96-u (T=32)",
          "dvsgc-81p": "DVS-GC 81-p (T=16)"}


def load_gap_scores(results_dir, datasets):
    """{(dataset, arm): {gap: [acc per seed]}} plus {dataset: chance}."""
    scores = defaultdict(lambda: defaultdict(list))
    chance = {}
    with open(os.path.join(results_dir, "summary.csv")) as f:
        for row in csv.DictReader(f):
            ds, arm = row["dataset"], row["arm"]
            if ds not in datasets or arm not in ARMS:
                continue
            if row["epochs_done"] != row["epochs_planned"]:
                continue
            path = os.path.join(results_dir, "runs", row["run"], "gap.json")
            if not os.path.exists(path):
                continue
            with open(path) as g:
                gap = json.load(g)
            chance[ds] = gap.get("chance")
            for k, acc in gap["gap"].items():
                scores[(ds, arm)][int(k)].append(acc)
    return scores, chance


def relative(by_gap):
    """{gap: [acc / gap-0 acc, per seed]}. Seeds line up because each run adds one value per gap."""
    base = by_gap[0]
    return {g: [a / b for a, b in zip(accs, base)] for g, accs in by_gap.items()}


def mean_std(vals):
    return statistics.mean(vals), (statistics.stdev(vals) if len(vals) > 1 else 0.0)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--results", default=RESULTS)
    p.add_argument("--datasets", nargs="+", default=["dvs128", "dvsgc-96u"])
    p.add_argument("--out", default=os.path.join(HERE, "figs", "gap_sweep.png"))
    args = p.parse_args()

    scores, chance = load_gap_scores(args.results, args.datasets)

    rows = []
    for (ds, arm), by_gap in sorted(scores.items()):
        rel = relative(by_gap)
        for gap in sorted(by_gap):
            m, s = mean_std(by_gap[gap])
            rm, rs = mean_std(rel[gap])
            rows.append({"dataset": ds, "arm": arm, "gap": gap, "n_seeds": len(by_gap[gap]),
                         "acc_mean": round(m, 4), "acc_std": round(s, 4),
                         "rel_mean": round(rm, 4), "rel_std": round(rs, 4)})
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(os.path.splitext(args.out)[0] + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "arm", "gap", "n_seeds", "acc_mean", "acc_std",
                                          "rel_mean", "rel_std"])
        w.writeheader()
        w.writerows(rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(args.datasets)
    fig, axes = plt.subplots(2, n, figsize=(5.2 * n, 8.0), sharex=True, sharey="row", squeeze=False)
    for col, ds in enumerate(args.datasets):
        top, bot = axes[0][col], axes[1][col]
        for i, (arm, label) in enumerate(ARMS.items()):
            by_gap = scores.get((ds, arm))
            if not by_gap:
                continue
            gaps = sorted(by_gap)
            n_seeds = min(len(by_gap[g]) for g in gaps)
            for ax, data in ((top, by_gap), (bot, relative(by_gap))):
                ms = [mean_std(data[g]) for g in gaps]
                ax.errorbar(gaps, [m for m, _ in ms], yerr=[s for _, s in ms], marker="o",
                            capsize=3, color=f"C{i}", label=f"{label} (n={n_seeds})")
        if chance.get(ds):
            top.axhline(chance[ds], color="gray", ls="--", lw=1, label="chance")
        bot.axhline(1.0, color="gray", ls=":", lw=1)
        top.set_title(TITLES.get(ds, ds))
        top.set_ylim(0, 1)
        bot.set_ylim(0, 1.1)
        bot.set_xlabel("empty frames inserted mid-clip")
        for ax in (top, bot):
            ax.set_xticks([0, 2, 4, 8, 16])
            ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("test accuracy (final epoch)")
    axes[1][0].set_ylabel("accuracy / gap-0 accuracy")
    axes[0][-1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

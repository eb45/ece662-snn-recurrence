"""
Copy the small result files from finished runs into results/ and rebuild the tables.

    python collect_results.py /work/eb408/ece662_runs/*/ /work/<august>/ece662_runs/*/

For each run folder with a final.json, this copies final.json, metrics.jsonl,
gap.json, and spikes.json (whichever exist) into results/runs/<run name>/.
Checkpoints are never copied. Runs without final.json (still training or timed
out) are skipped.

Then it rebuilds, from everything in results/runs/ (not only the folders you
passed, so you can collect a few at a time):
  results/summary.csv     one row per run
  results/main_table.csv  every planned (dataset, arm), A to E plus LTC: final accuracy per seed,
                          mean and std, and who still owes which seed (full 64-epoch runs only)
  results/grid_table.csv  DVS128 seed-2026 placement x W_rec setting grid, blank cells to run
"""

import argparse
import csv
import json
import os
import shutil
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
RUNS = os.path.join(RESULTS, "runs")
KEEP = ("final.json", "metrics.jsonl", "gap.json", "spikes.json")
GAPS = (0, 2, 4, 8, 16)
MIN_EPOCHS = 64  # runs shorter than this stay in summary.csv but not in main_table.csv


def read_json(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def read_metrics(path):
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    # A requeued run can append an epoch twice; keep the last copy of each.
    by_epoch = {r["epoch"]: r for r in rows}
    return [by_epoch[e] for e in sorted(by_epoch)]


def dataset_variant(args):
    """dvs128, dvsgc-81p, dvsgc-96u, or dvsgc-<custom> for anything else."""
    if args.get("dataset", "dvs128") == "dvs128":
        return "dvs128"
    # Runs from before the 96-u flags existed have no alpha/repeat args: they used the 81-p defaults.
    seq, cls = args.get("seq_len", 4), args.get("class_num", 3)
    lo, hi = args.get("alpha_min", 0.5), args.get("alpha_max", 0.7)
    no_rep = args.get("no_repeat", False)
    if (seq, cls, lo, hi, no_rep) == (4, 3, 0.5, 0.7, False):
        return "dvsgc-81p"
    if (seq, cls, lo, hi, no_rep) == (6, 3, 0.2, 1.0, True):
        return "dvsgc-96u"
    return f"dvsgc-L{seq}c{cls}a{lo}-{hi}{'nr' if no_rep else ''}"


def describe(args):
    """(model, placement, how W_rec learns, arm letter or '')."""
    if "ltc" in (str(args.get("neuron", "")) + str(args.get("model", "")) + str(args.get("out", ""))).lower():
        return "ltc", "", "LTC neurons", "LTC"
    layers = args.get("rec_layers")
    if not layers:
        neuron = args.get("neuron", "lif")
        return ("control" if neuron == "lif" else f"control-{neuron}"), "", "", \
               ("A" if neuron == "lif" else "")
    place = "+".join(layers)
    lr, wd, anchor = args.get("rec_lr"), args.get("rec_wd", 0.0) or 0.0, args.get("rec_anchor", False)
    if args.get("freeze_rec"):
        how = "frozen"
    elif anchor and wd > 0:
        how = f"anchored wd {wd:g}"
    elif wd > 0:
        how = f"plain wd {wd:g}"
    elif lr is not None and lr != args.get("lr"):
        how = f"lr {lr:g}"
    else:
        how = f"lr {args.get('lr', 1e-3):g}"
    if args.get("readout") == "late":
        how += ", late readout"

    arm = ""
    if place == "fc" and args.get("readout", "mean") == "mean" and not args.get("clip_grad"):
        if how == "frozen":
            arm = "B"
        elif how == f"lr {args.get('lr', 1e-3):g}" and args.get("rec_init", "orthogonal") == "orthogonal":
            arm = "C"
        elif how == "lr 0.0001":
            arm = "D"
        elif how == "anchored wd 1":
            arm = "E"
    return "recurrent", place, how, arm


def summarize(run_dir):
    final = read_json(os.path.join(run_dir, "final.json"))
    args = final["args"]
    metrics = read_metrics(os.path.join(run_dir, "metrics.jsonl"))
    gap = read_json(os.path.join(run_dir, "gap.json")) or {}
    spikes = read_json(os.path.join(run_dir, "spikes.json")) or {}
    model, place, how, arm = describe(args)

    last = metrics[-1] if metrics else {}
    final_acc = final.get("final_test_acc", last.get("test_acc"))
    drift = ((final.get("rec_final_stats") or {}).get("all") or {})
    secs = [m["epoch_sec"] for m in metrics if "epoch_sec" in m]

    row = {
        "run": os.path.basename(run_dir.rstrip("/")),
        "dataset": dataset_variant(args),
        "arm": arm,
        "model": model,
        "w_rec_placement": place,
        "w_rec_learning": how,
        "seed": args.get("seed"),
        "T": args.get("T"),
        "epochs_done": len(metrics),
        "epochs_planned": args.get("epochs"),
        "final_test_acc": final_acc,
        "best_test_acc": final.get("best_test_acc"),
        "final_train_acc": last.get("train_acc"),
        "w_rec_drift_rel": drift.get("drift_rel"),
        "w_rec_spec_norm_max": drift.get("spec_norm_max"),
        "params": final.get("params"),
        "peak_gpu_gb": max((m.get("peak_gpu_gb", 0) for m in metrics), default=None),
        "median_epoch_sec": statistics.median(secs) if secs else None,
    }
    for g in GAPS:
        row[f"gap{g}_acc"] = (gap.get("gap") or {}).get(str(g))
    row["reverse_acc"] = gap.get("reverse")
    row["shuffle_acc"] = gap.get("shuffle_mean")
    for k, v in spikes.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            row[f"spk_{k}"] = v
    return row


def write_csv(path, rows):
    if not rows:
        return
    cols = []
    for r in rows:
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r[k]) for k in cols})


# Every run the paper needs, so missing ones show up as blank cells with an owner.
# (dataset, arm) -> {seed: owner}
SEEDS = (2026, 1, 2)
PLAN = {}
for _arm in "ABCDE":
    PLAN[("dvs128", _arm)] = {2026: "August", 1: "August", 2: "August"}
    PLAN[("dvsgc-96u", _arm)] = {2026: "Emma", 1: "August", 2: "August"}
PLAN[("dvs128", "A")][2026] = "Emma"
for _arm in "ABC":
    PLAN[("dvsgc-81p", _arm)] = {2026: "Emma"}
PLAN[("dvs128", "LTC")] = {2026: "Anna", 1: "Anna", 2: "Anna"}
ARM_NAMES = {"A": "control", "B": "frozen W_rec", "C": "learned W_rec (lr 1e-3)",
             "D": "slow W_rec (lr 1e-4)", "E": "anchored W_rec (wd 1.0)", "LTC": "LTC neurons"}
DATASET_ORDER = {"dvs128": 0, "dvsgc-81p": 1, "dvsgc-96u": 2}

# Placement x learning-setting grid on DVS128, seed 2026 (August).
GRID_PLACEMENTS = ("fc", "conv5", "conv5+fc")
GRID_SETTINGS = ("frozen", "lr 3e-05", "lr 0.0001", "lr 0.0003", "lr 0.001", "anchored wd 1")
GRID_FC_EXTRAS = ("anchored wd 0.3", "anchored wd 3", "plain wd 0.1", "plain wd 1")


def _full(r):
    """Only full-length runs count: debug and short test runs are left out."""
    return bool(r["epochs_planned"]) and r["epochs_done"] >= r["epochs_planned"] >= MIN_EPOCHS \
        and r["final_test_acc"] is not None


def _fmt(x):
    return "" if x is None else round(x, 4)


def main_table(rows):
    """Arms A to E (+ LTC) per dataset: one row per planned (dataset, arm), filled where done.

    Per-seed final accuracy, then mean and std over the seeds that are done.
    Arms that ran but are not in PLAN are added at the end.
    """
    done = {}
    for r in rows:
        if r["arm"] and _full(r):
            done.setdefault((r["dataset"], r["arm"]), {})[r["seed"]] = r
    keys = list(PLAN) + [k for k in done if k not in PLAN]
    keys.sort(key=lambda k: (DATASET_ORDER.get(k[0], 9), "ABCDE".find(k[1]) % 10 if k[1] in "ABCDE" else 8))
    out = []
    for ds, arm in keys:
        plan = PLAN.get((ds, arm), {})
        runs = done.get((ds, arm), {})
        accs = [runs[s]["final_test_acc"] for s in SEEDS if s in runs]
        gap8 = [runs[s]["gap8_acc"] for s in SEEDS if s in runs and runs[s]["gap8_acc"] is not None]
        missing = [s for s in plan if s not in runs]
        row = {"dataset": ds, "arm": arm, "model": ARM_NAMES.get(arm, arm),
               "seeds_done": f"{len(runs)}/{len(plan) or len(runs)}"}
        for s in SEEDS:
            if s in runs:
                row[f"final_s{s}"] = _fmt(runs[s]["final_test_acc"])
            elif s in plan:
                row[f"final_s{s}"] = f"to run ({plan[s]})"
            else:
                row[f"final_s{s}"] = "-"
        row["final_acc_mean"] = _fmt(statistics.mean(accs)) if accs else ""
        row["final_acc_std"] = _fmt(statistics.stdev(accs)) if len(accs) > 1 else ""
        row["gap8_acc_mean"] = _fmt(statistics.mean(gap8)) if gap8 else ""
        row["drift_mean"] = _fmt(statistics.mean(
            [runs[s]["w_rec_drift_rel"] for s in runs if runs[s]["w_rec_drift_rel"] is not None])) \
            if any(runs[s]["w_rec_drift_rel"] is not None for s in runs) else ""
        row["missing"] = ", ".join(f"s{s} ({plan[s]})" for s in missing)
        out.append(row)
    return out


def grid_table(rows):
    """DVS128 seed-2026 placement x W_rec setting grid: final accuracy and drift per cell."""
    cell = {}
    for r in rows:
        if r["dataset"] == "dvs128" and r["seed"] == 2026 and r["model"] == "recurrent" and _full(r):
            cell[(r["w_rec_placement"], r["w_rec_learning"])] = r
    out = []
    for place in GRID_PLACEMENTS:
        settings = GRID_SETTINGS + (GRID_FC_EXTRAS if place == "fc" else ())
        for how in settings:
            r = cell.get((place, how))
            out.append({
                "placement": place, "w_rec_learning": how,
                "final_test_acc": _fmt(r["final_test_acc"]) if r else "to run (August)",
                "best_test_acc": _fmt(r["best_test_acc"]) if r else "",
                "final_train_acc": _fmt(r["final_train_acc"]) if r else "",
                "w_rec_drift_rel": _fmt(r["w_rec_drift_rel"]) if r else "",
                "w_rec_spec_norm_max": _fmt(r["w_rec_spec_norm_max"]) if r else "",
                "run": r["run"] if r else "",
            })
    return out


def collect(run_dirs):
    os.makedirs(RUNS, exist_ok=True)
    for d in run_dirs:
        if not os.path.exists(os.path.join(d, "final.json")):
            print(f"skip {d} (no final.json yet)")
            continue
        dest = os.path.join(RUNS, os.path.basename(d.rstrip("/")))
        os.makedirs(dest, exist_ok=True)
        copied = []
        for name in KEEP:
            src = os.path.join(d, name)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dest, name))
                copied.append(name)
        print(f"{os.path.basename(dest)}: {', '.join(copied)}")


def rebuild():
    names = sorted(n for n in os.listdir(RUNS)
                   if os.path.exists(os.path.join(RUNS, n, "final.json"))) if os.path.isdir(RUNS) else []
    rows = [summarize(os.path.join(RUNS, n)) for n in names]
    rows.sort(key=lambda r: (r["dataset"], r["arm"] or "Z", r["model"], r["w_rec_placement"],
                             r["w_rec_learning"], r["seed"] or 0))
    write_csv(os.path.join(RESULTS, "summary.csv"), rows)
    write_csv(os.path.join(RESULTS, "main_table.csv"), main_table(rows))
    write_csv(os.path.join(RESULTS, "grid_table.csv"), grid_table(rows))
    print(f"summary.csv: {len(rows)} runs")
    return rows


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="*", help="Run folders to copy in (omit to just rebuild the tables)")
    args = p.parse_args(argv)
    collect(args.runs)
    rebuild()


if __name__ == "__main__":
    main()

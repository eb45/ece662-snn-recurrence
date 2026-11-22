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
  results/main_table.csv  arms A to E: final-epoch mean and std over seeds, per dataset
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
    seq, cls = args.get("seq_len"), args.get("class_num")
    lo, hi = args.get("alpha_min"), args.get("alpha_max")
    no_rep = args.get("no_repeat", False)
    if (seq, cls, lo, hi, no_rep) == (4, 3, 0.5, 0.7, False):
        return "dvsgc-81p"
    if (seq, cls, lo, hi, no_rep) == (6, 3, 0.2, 1.0, True):
        return "dvsgc-96u"
    return f"dvsgc-L{seq}c{cls}a{lo}-{hi}{'nr' if no_rep else ''}"


def describe(args):
    """(model, placement, how W_rec learns, arm letter or '')."""
    if args.get("neuron") == "ltc" or args.get("model") == "ltc":
        return "ltc", "", "", ""
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


def main_table(rows):
    """Arms A to E, grouped by dataset: final-epoch accuracy mean and std over seeds."""
    groups = {}
    for r in rows:
        if r["arm"] and r["final_test_acc"] is not None:
            groups.setdefault((r["dataset"], r["arm"]), []).append(r)
    out = []
    for (ds, arm), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        accs = [r["final_test_acc"] for r in rs]
        out.append({
            "dataset": ds, "arm": arm,
            "w_rec_learning": rs[0]["w_rec_learning"] or "none",
            "n_seeds": len(accs), "seeds": " ".join(str(r["seed"]) for r in rs),
            "final_acc_mean": round(statistics.mean(accs), 4),
            "final_acc_std": round(statistics.stdev(accs), 4) if len(accs) > 1 else "",
            "best_acc_mean": round(statistics.mean(r["best_test_acc"] for r in rs), 4),
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

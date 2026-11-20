"""
Time-shuffle test for a finished feed-forward control run.

Loads <run>/best.pt, scores it on the clean test set, then again with the time
bins of every test clip randomly permuted (datasets_extra.TimeShuffle), once
per seed. A large drop means the model relies on temporal order.

Reads the dataset and model settings from <run>/final.json, so the run must
have finished. Writes <run>/shuffle.json.

    python eval_shuffle.py --run /work/eb408/ece662_runs/dvsgc_ff_s2026
"""

import argparse
import json
import os
import statistics

import torch

from datasets_extra import TimeShuffle, get_dataset_loaders
from evaluate import evaluate
from models import CSNN


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Time-shuffle test for a finished control run.")
    p.add_argument("--run", required=True, help="Run folder with best.pt and final.json")
    p.add_argument("--root", default=None, help="Override the data root stored in final.json")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    with open(os.path.join(args.run, "final.json")) as f:
        final = json.load(f)
    run_args = final["args"]
    dataset = run_args.get("dataset", "dvs128")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = (not args.no_amp) and device.type == "cuda"

    dataset_kwargs = {}
    if dataset == "dvsgc":
        dataset_kwargs = dict(seq_len=run_args["seq_len"], class_num=run_args["class_num"])
    _, test_loader, info = get_dataset_loaders(
        dataset, args.root or run_args["root"], batch_size=args.batch_size, T=run_args["T"],
        num_workers=args.num_workers, seed=run_args["seed"], **dataset_kwargs)

    num_classes = info["num_classes"]
    in_channels, input_size, _ = info["frame_shape"]
    model = CSNN(channels=run_args["channels"], num_classes=num_classes,
                 in_channels=in_channels, input_size=input_size).to(device)
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pt"), map_location=device))

    clean = evaluate(model, test_loader, device, amp=use_amp)
    print(f"dataset: {dataset}   T: {run_args['T']}   chance: {1 / num_classes:.4f}")
    print(f"clean test acc: {clean:.4f}")

    shuffled = []
    for s in args.seeds:
        acc = evaluate(model, test_loader, device, transform=TimeShuffle(seed=s), amp=use_amp)
        shuffled.append(acc)
        print(f"shuffled (seed {s}): {acc:.4f}")

    mean = statistics.mean(shuffled)
    std = statistics.stdev(shuffled) if len(shuffled) > 1 else 0.0
    result = {
        "dataset": dataset, "T": run_args["T"], "num_classes": num_classes,
        "chance": 1 / num_classes, "clean_acc": clean,
        "shuffled_acc": shuffled, "shuffled_mean": mean, "shuffled_std": std,
        "drop_abs": clean - mean,
        "drop_rel": (clean - mean) / clean if clean > 0 else None,
        "seeds": args.seeds,
    }
    with open(os.path.join(args.run, "shuffle.json"), "w") as f:
        json.dump(result, f, indent=2)
    rel = f"{result['drop_rel']:.1%}" if result["drop_rel"] is not None else "n/a"
    print(f"shuffled mean: {mean:.4f} +/- {std:.4f}   drop: {clean - mean:.4f} ({rel} of clean)")
    return result


if __name__ == "__main__":
    main()

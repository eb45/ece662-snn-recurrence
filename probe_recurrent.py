"""
Diagnostics for trained recurrent-SNN runs (and the control).

For each run it reports, on the test split:
  - top-1 accuracy with the readout the run was trained with (should match
    best_test_acc in final.json, up to a clip or so from mixed precision)
  - firing rate of every spiking layer (fraction of neuron-steps that spike)
  - spikes per clip, per layer and in total (efficiency proxy)
  - for each recurrent layer, RMS of the recurrent input W_rec * s_{t-1}
    relative to the feed-forward input BN(Conv(x_t)), i.e. how much the
    layer leans on recurrence

    python probe_recurrent.py --root /hpc/home/ah670/dvs128_gesture \
        --runs /work/ah670/ece662_runs/rec_fc_s2026 /work/ah670/ece662_runs/rec_conv5_fc_frozen_s2026

Runs whose final.json has no rec_layers are loaded as the control (models.CSNN).
Results are printed and written to <run>/probe.json.
"""

import argparse
import json
import os
from collections import defaultdict

import torch
from spikingjelly.activation_based import functional, neuron

from dvs_data import DEFAULT_CONFIG, get_loaders
from models import CSNN
from recurrent_model import LateReadout, RecurrentLIF, load_run


def load_any(run_dir: str, device):
    with open(os.path.join(run_dir, "final.json")) as f:
        args = json.load(f)["args"]
    if "rec_layers" in args:
        return load_run(run_dir, device)
    net = CSNN(channels=args["channels"], num_classes=DEFAULT_CONFIG["num_classes"])
    net.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), map_location=device))
    return net.to(device).eval()


def layer_label(name: str) -> str:
    # conv_stack.{4(b-1)+2}[.lif] -> conv{b};  head.3[.lif] -> fc;  head.6 -> out
    parts = name.split(".")
    if parts[0] == "conv_stack":
        return f"conv{int(parts[1]) // 4 + 1}"
    return {"3": "fc", "6": "out"}.get(parts[1], name)


@torch.no_grad()
def probe(model, loader, device, max_batches=None) -> dict:
    net = model.model if isinstance(model, LateReadout) else model
    model.eval()

    # Sums are kept as tensors on the device so the hooks never force a sync.
    stats = defaultdict(lambda: defaultdict(float))
    handles = []
    for name, m in net.named_modules():
        if isinstance(m, neuron.BaseNode):
            label = layer_label(name)

            def count(mod, inp, out, label=label):
                stats[label]["spikes"] = stats[label]["spikes"] + out.float().sum()
                stats[label]["elems"] += out.numel()
            handles.append(m.register_forward_hook(count))
        elif isinstance(m, RecurrentLIF):
            label = layer_label(name)

            def ff(mod, inp, label=label):
                x = inp[0].float()
                stats[label]["ff_sq"] = stats[label]["ff_sq"] + x.pow(2).sum()
                stats[label]["ff_n"] += x.numel()

            def rec(mod, inp, out, label=label):
                r = out.float()
                stats[label]["rec_sq"] = stats[label]["rec_sq"] + r.pow(2).sum()
                stats[label]["rec_n"] += r.numel()
            handles.append(m.register_forward_pre_hook(ff))
            handles.append(m.rec.register_forward_hook(rec))

    correct, total = 0, 0
    try:
        for i, (frames, labels) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            out = model(frames).mean(0)
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.numel()
            functional.reset_net(model)
    finally:
        for h in handles:
            h.remove()

    layers = {}
    for label, s in stats.items():
        spikes = float(s["spikes"])
        row = {"firing_rate": spikes / s["elems"], "spikes_per_clip": spikes / total}
        if s["rec_n"]:
            # rec_rms includes t = 0, where s_{-1} = 0, so it slightly understates
            # the recurrent drive once the layer is running.
            ff_rms = (float(s["ff_sq"]) / s["ff_n"]) ** 0.5
            rec_rms = (float(s["rec_sq"]) / s["rec_n"]) ** 0.5
            row.update(ff_rms=ff_rms, rec_rms=rec_rms,
                       rec_to_ff=rec_rms / ff_rms if ff_rms > 0 else float("nan"))
        layers[label] = row

    return {"accuracy": correct / total, "clips": total,
            "spikes_per_clip": sum(r["spikes_per_clip"] for r in layers.values()),
            "layers": layers}


def print_report(run_dir: str, res: dict) -> None:
    print(f"\n{os.path.basename(os.path.normpath(run_dir))}: acc {res['accuracy']:.4f} "
          f"on {res['clips']} clips, {res['spikes_per_clip']:,.0f} spikes/clip")
    print(f"  {'layer':6} {'rate':>7} {'spikes/clip':>13} {'rec/ff':>7}")
    for label, r in res["layers"].items():
        ratio = f"{r['rec_to_ff']:.3f}" if "rec_to_ff" in r else "-"
        print(f"  {label:6} {r['firing_rate']:7.4f} {r['spikes_per_clip']:13,.0f} {ratio:>7}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="DVS128 Gesture root (contains download/)")
    p.add_argument("--runs", nargs="+", required=True, help="Run folders containing final.json")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-batches", type=int, default=None, help="Probe only the first N test batches")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_loaders(args.root, batch_size=args.batch_size, num_workers=args.num_workers)

    for run in args.runs:
        res = probe(load_any(run, device), test_loader, device, args.max_batches)
        print_report(run, res)
        with open(os.path.join(run, "probe.json"), "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()

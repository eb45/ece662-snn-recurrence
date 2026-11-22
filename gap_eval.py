"""
Gap and time-order tests for finished runs (control or recurrent, any dataset).

For each run folder it loads the model, then scores the test set:
  * clean
  * with g empty frames inserted mid-clip, for each g in --gaps
  * with the time bins reversed
  * with the time bins shuffled (one score per --shuffle-seeds, averaged)
and writes <run>/gap.json. No retraining.

    python gap_eval.py --runs /work/$USER/ece662_runs/control_s2026 \
                              /work/$USER/ece662_runs/rec_fc_s2026

By default it scores checkpoint.pt (the last epoch), since the report uses
final-epoch numbers. --which best uses best.pt instead.

--bn-check also records how often conv1 neurons fire during the inserted empty
frames vs during real frames. Empty frames are zeros, but BatchNorm in eval mode
turns zero into a constant per-channel current, so the gap may not be silent.
"""

import argparse
import json
import os
import statistics

import torch
from spikingjelly.activation_based import functional

from evaluate import evaluate
from models import CSNN


# ----------------------------------------------------------------- transforms

def insert_gap(frames: torch.Tensor, start: int, length: int) -> torch.Tensor:
    """Insert `length` all-zero frames before time step `start`.

    frames is time-first [T, B, ...]. Returns [T + length, B, ...]. Every
    original event is kept; only the delay between part A (steps < start)
    and part B (steps >= start) grows.
    """
    if length <= 0:
        return frames
    T = frames.shape[0]
    if not 0 <= start <= T:
        raise ValueError(f"start must be in [0, {T}], got {start}")
    gap = frames.new_zeros((length, *frames.shape[1:]))
    return torch.cat([frames[:start], gap, frames[start:]], dim=0)


def reverse_time(frames: torch.Tensor) -> torch.Tensor:
    """Play every clip backwards."""
    return frames.flip(0)


class TimeShuffle:
    """Randomly permute the time bins of each clip (same as datasets_extra.TimeShuffle)."""

    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        T, B = frames.shape[:2]
        perm = torch.rand(T, B, generator=self.gen).argsort(dim=0).to(frames.device)
        idx = perm.view(T, B, *([1] * (frames.dim() - 2))).expand_as(frames)
        return frames.gather(0, idx)


# ------------------------------------------------------------ run loading

def load_model(run_dir: str, num_classes: int, frame_shape, device, which: str = "final"):
    """Rebuild a control or recurrent run from final.json and load its weights."""
    with open(os.path.join(run_dir, "final.json")) as f:
        args = json.load(f)["args"]
    in_channels, input_size = frame_shape[0], frame_shape[1]

    if args.get("rec_layers"):
        from recurrent_model import LateReadout, RecurrentCSNN
        net = RecurrentCSNN(rec_layers=args["rec_layers"], rec_init=args["rec_init"],
                            rec_gain=args["rec_gain"], freeze_rec=args["freeze_rec"],
                            channels=args["channels"], num_classes=num_classes,
                            in_channels=in_channels, input_size=input_size)
    else:
        net = CSNN(channels=args["channels"], num_classes=num_classes,
                   in_channels=in_channels, input_size=input_size,
                   neuron=args.get("neuron", "lif"))

    fname = "checkpoint.pt" if which == "final" else "best.pt"
    state = torch.load(os.path.join(run_dir, fname), map_location=device)
    if isinstance(state, dict) and "model" in state and "opt" in state:
        state = state["model"]
    net.load_state_dict(state)
    net.to(device).eval()

    if args.get("readout") == "late":
        return LateReadout(net, args["late_k"]), args
    return net, args


def get_test_loader(run_args: dict, root: str, batch_size: int, num_workers: int):
    """Test loader for whichever dataset the run was trained on."""
    dataset = run_args.get("dataset", "dvs128")
    T, seed = run_args["T"], run_args["seed"]

    if dataset == "dvs128":
        from dvs_data import DEFAULT_CONFIG, get_loaders
        _, test_loader = get_loaders(root, batch_size=batch_size, T=T,
                                     num_workers=num_workers, seed=seed)
        return test_loader, DEFAULT_CONFIG["num_classes"], (2, 128, 128)

    from datasets_extra import get_dataset_loaders
    keys = ("seq_len", "class_num", "alpha_min", "alpha_max", "repeat")
    kwargs = {k: run_args[k] for k in keys if k in run_args}
    if run_args.get("no_repeat"):
        kwargs["repeat"] = False
    _, test_loader, info = get_dataset_loaders(dataset, root, batch_size=batch_size, T=T,
                                               num_workers=num_workers, seed=seed, **kwargs)
    return test_loader, info["num_classes"], tuple(info["frame_shape"])


# ------------------------------------------------------------ BatchNorm check

@torch.no_grad()
def bn_gap_firing(model, loader, device, start: int, length: int, max_batches: int = 10):
    """Mean conv1 firing rate during inserted empty frames vs during real frames."""
    net = model.model if hasattr(model, "model") else model  # unwrap LateReadout
    lif = net.conv_stack[2]  # conv block 1 is [Conv, BN, LIF, Pool]
    store = {}
    handle = lif.register_forward_hook(lambda m, i, o: store.__setitem__("s", o))
    gap_sum, real_sum, n = 0.0, 0.0, 0
    try:
        for b, (frames, _) in enumerate(loader):
            if b >= max_batches:
                break
            x = insert_gap(frames.to(device), start, length)
            net(x)
            s = store["s"].float()  # [T + length, B, C, H, W]
            gap_steps = s[start:start + length]
            real_steps = torch.cat([s[:start], s[start + length:]], dim=0)
            gap_sum += gap_steps.mean().item()
            real_sum += real_steps.mean().item()
            n += 1
            functional.reset_net(net)
    finally:
        handle.remove()
    return {"gap_length": length, "conv1_rate_gap": gap_sum / n, "conv1_rate_real": real_sum / n}


# ------------------------------------------------------------ main

def run_tests(model, loader, device, T: int, gaps, gap_pos: float, shuffle_seeds, amp: bool):
    start = int(round(T * gap_pos))
    out = {"T": T, "gap_start": start}
    out["clean"] = evaluate(model, loader, device, amp=amp)
    out["gap"] = {}
    for g in gaps:
        out["gap"][str(g)] = (out["clean"] if g == 0 else
                              evaluate(model, loader, device, amp=amp,
                                       transform=lambda x, g=g: insert_gap(x, start, g)))
    out["reverse"] = evaluate(model, loader, device, transform=reverse_time, amp=amp)
    shuf = [evaluate(model, loader, device, transform=TimeShuffle(s), amp=amp)
            for s in shuffle_seeds]
    out["shuffle"] = shuf
    out["shuffle_mean"] = statistics.mean(shuf) if shuf else None
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs", nargs="+", required=True, help="Run folders with final.json")
    p.add_argument("--root", default=None, help="Override the data root saved in final.json")
    p.add_argument("--which", choices=["final", "best"], default="final")
    p.add_argument("--gaps", type=int, nargs="+", default=[0, 2, 4, 8, 16])
    p.add_argument("--gap-pos", type=float, default=0.5, help="Where the gap goes, as a fraction of T")
    p.add_argument("--shuffle-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--bn-check", action="store_true")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (not args.no_amp) and device.type == "cuda"

    for run in args.runs:
        with open(os.path.join(run, "final.json")) as f:
            run_args = json.load(f)["args"]
        loader, num_classes, frame_shape = get_test_loader(
            run_args, args.root or run_args["root"], args.batch_size, args.num_workers)
        model, _ = load_model(run, num_classes, frame_shape, device, args.which)

        result = run_tests(model, loader, device, run_args["T"], args.gaps, args.gap_pos,
                           args.shuffle_seeds, amp)
        result.update(run=run, dataset=run_args.get("dataset", "dvs128"),
                      seed=run_args["seed"], checkpoint=args.which,
                      num_classes=num_classes, chance=1 / num_classes)
        if args.bn_check:
            result["bn_check"] = bn_gap_firing(model, loader, device, result["gap_start"], 8)

        with open(os.path.join(run, "gap.json"), "w") as f:
            json.dump(result, f, indent=2)
        gaps = "  ".join(f"g{g}={result['gap'][str(g)]:.3f}" for g in args.gaps)
        print(f"{os.path.basename(run.rstrip('/'))}: clean {result['clean']:.3f}  {gaps}  "
              f"reverse {result['reverse']:.3f}  shuffle {result['shuffle_mean']:.3f}", flush=True)


if __name__ == "__main__":
    main()

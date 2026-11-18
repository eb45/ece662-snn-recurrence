"""
Train the deep recurrent SNN on DVS128 Gesture.

Same data, optimizer, loss, schedule, batching, checkpointing, and scoring as
train_control.py. The differences are the model (recurrent_model.RecurrentCSNN)
and the recurrence/readout flags below; see README_RECURRENT.md.

Quick debug run (2 epochs) to check shapes, memory, and time per epoch:
    python train_recurrent.py --root /work/eb408/dvs128_gesture --out /work/eb408/ece662_runs/rec_debug --epochs 2

Main arm (learned recurrence at block 5 and the hidden FC layer):
    python train_recurrent.py --root ... --out .../rec_conv5_fc_s2026 --rec-layers conv5 fc

Random fixed recurrence (same network, W_rec frozen at its random init):
    python train_recurrent.py --root ... --out .../rec_conv5_fc_frozen_s2026 --rec-layers conv5 fc --freeze-rec

Late readout (score only the last k steps; for the memory experiments):
    python train_recurrent.py --root ... --out .../rec_conv5_fc_late_s2026 --readout late --late-k 4

Regularize W_rec (learned recurrence overfits the training people otherwise):
    python train_recurrent.py --root ... --out .../rec_fc_wd1_s2026 --rec-layers fc --rec-wd 1.0

Checkpoints are written every epoch and the script resumes automatically
if <out>/checkpoint.pt exists, so a preempted scavenger job can requeue and
carry on.
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from dvs_data import DEFAULT_CONFIG, get_loaders, set_seed
from evaluate import evaluate
from recurrent_model import REC_INITS, REC_LAYER_NAMES, LateReadout, RecurrentCSNN


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True, help="DVS128 Gesture root (contains download/)")
    p.add_argument("--out", required=True, help="Output folder for checkpoints and metrics")
    p.add_argument("--epochs", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-step batch size. 8 is a safe start for an 11 GB GPU.")
    p.add_argument("--accum", type=int, default=2,
                   help="Gradient accumulation steps. Effective batch = batch-size * accum.")
    p.add_argument("--T", type=int, default=DEFAULT_CONFIG["T"])
    p.add_argument("--channels", type=int, default=128)
    p.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--loss", choices=["mse", "ce"], default="mse",
                   help="mse on firing rate vs one-hot is what the tutorial uses")
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--no-resume", action="store_true")
    # Recurrent-arm options
    p.add_argument("--rec-layers", nargs="+", choices=REC_LAYER_NAMES, default=["conv5", "fc"],
                   help="LIF layers that get trainable recurrent weights")
    p.add_argument("--rec-init", choices=REC_INITS, default="orthogonal")
    p.add_argument("--rec-gain", type=float, default=0.5,
                   help="Scale of the orthogonal W_rec init (spectral radius for fc)")
    p.add_argument("--freeze-rec", action="store_true",
                   help="Keep W_rec at its random init (random-recurrence arm)")
    p.add_argument("--readout", choices=["mean", "late"], default="mean",
                   help="mean: average output over all steps (control). late: last --late-k steps only")
    p.add_argument("--late-k", type=int, default=4)
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="Max grad norm; 0 disables (control does not clip)")
    p.add_argument("--rec-lr", type=float, default=None,
                   help="Learning rate for W_rec only (default: same as --lr)")
    p.add_argument("--rec-wd", type=float, default=0.0,
                   help="Decoupled (AdamW-style) weight decay on W_rec only; 0 disables")
    return p.parse_args()


def build_optimizer(net, args):
    """Optimizer as in the control; W_rec can get its own lr and weight decay.

    Returns (optimizer, trainable params) so the caller can clip the same params.
    """
    trainable = [p for p in net.parameters() if p.requires_grad]
    rec_ids = {id(m.rec.weight) for m in net.rec_modules().values()}
    rec_trainable = [p for p in trainable if id(p) in rec_ids]
    if rec_trainable and (args.rec_lr is not None or args.rec_wd > 0):
        # Everything except W_rec is optimized exactly as in the control (no decay).
        groups = [
            {"params": [p for p in trainable if id(p) not in rec_ids], "weight_decay": 0.0},
            {"params": rec_trainable, "lr": args.rec_lr or args.lr, "weight_decay": args.rec_wd},
        ]
        print(f"W_rec optimizer group: lr={args.rec_lr or args.lr}, weight_decay={args.rec_wd}")
    else:
        groups = [{"params": trainable, "weight_decay": 0.0}]
        if args.rec_lr is not None or args.rec_wd > 0:
            print("WARNING: --rec-lr/--rec-wd ignored because W_rec is frozen")
    if args.opt == "adam":
        # AdamW with weight_decay=0 is exactly Adam, so the non-recurrent weights
        # train as in the control. The W_rec decay is decoupled from Adam's
        # per-parameter scaling (plain Adam L2 decay would be scaled away).
        opt = torch.optim.AdamW(groups, lr=args.lr)
    else:
        opt = torch.optim.SGD(groups, lr=args.lr, momentum=0.9)
    return opt, trainable


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found, training will be very slow")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_loader, test_loader = get_loaders(
        args.root, batch_size=args.batch_size, T=args.T,
        num_workers=args.num_workers, seed=args.seed,
    )

    net = RecurrentCSNN(
        rec_layers=args.rec_layers, rec_init=args.rec_init, rec_gain=args.rec_gain,
        freeze_rec=args.freeze_rec, channels=args.channels,
        num_classes=DEFAULT_CONFIG["num_classes"],
    ).to(device)
    # `model` is what the loss and evaluate() see; `net` is what gets saved.
    model = LateReadout(net, args.late_k) if args.readout == "late" else net

    n_params, n_lif = net.num_params(), net.num_lif_neurons()
    n_total, n_rec = net.num_total_params(), net.num_rec_params()
    print(f"params: {n_params:,} trainable / {n_total:,} total ({n_rec:,} recurrent"
          f"{', frozen' if args.freeze_rec else ''})   LIF neurons per step: {n_lif:,}")
    print(f"recurrent layers: {' '.join(net.rec_layers)}   init: {args.rec_init} x {args.rec_gain}   "
          f"readout: {args.readout}{f' (last {args.late_k})' if args.readout == 'late' else ''}")
    print(f"batch {args.batch_size} x accum {args.accum} = effective {args.batch_size * args.accum}, "
          f"T={args.T}, amp={use_amp}, opt={args.opt}, lr={args.lr}, loss={args.loss}, "
          f"clip={args.clip_grad or 'off'}")

    opt, trainable = build_optimizer(net, args)
    sched =torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    best_path = os.path.join(args.out, "best.pt")
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    start_epoch, best_acc, last_acc = 0, 0.0, None

    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device)
        net.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
        last_acc = ck.get("last_acc")
        print(f"Resumed from epoch {start_epoch} (best test acc so far {best_acc:.4f})")

    num_classes = DEFAULT_CONFIG["num_classes"]

    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        run_loss, run_correct, run_total = 0.0, 0, 0
        opt.zero_grad(set_to_none=True)
        n_batches = len(train_loader)

        for i, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(frames).mean(0)  # firing rate, [B, num_classes]
            out = out.float()
            if args.loss == "mse":
                loss = F.mse_loss(out, F.one_hot(labels, num_classes).float())
            else:
                loss = F.cross_entropy(out, labels)

            # Recurrent BPTT is the arm most likely to blow up. Stop with a clear
            # message instead of silently training on NaNs.
            if not math.isfinite(loss.item()):
                raise RuntimeError(f"non-finite loss at epoch {epoch}, batch {i}; "
                                   f"try --clip-grad 1.0 or a smaller --rec-gain")

            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == n_batches:
                if args.clip_grad > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(trainable, args.clip_grad)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            functional.reset_net(model)

            run_loss += loss.item() * labels.numel()
            run_correct += (out.argmax(1) == labels).sum().item()
            run_total += labels.numel()

        sched.step()
        train_time = time.time() - t0

        test_acc = evaluate(model, test_loader, device, amp=use_amp)
        last_acc = test_acc
        epoch_time = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

        if test_acc > best_acc:
            best_acc = test_acc
            save_atomic(net.state_dict(), best_path)

        row = {
            "epoch": epoch, "train_loss": run_loss / run_total,
            "train_acc": run_correct / run_total, "test_acc": test_acc,
            "best_test_acc": best_acc, "train_sec": round(train_time, 1),
            "epoch_sec": round(epoch_time, 1), "peak_gpu_gb": round(peak_gb, 2),
            "rec_weight_norm": net.rec_weight_norms(),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"train {row['train_acc']:.3f}  test {test_acc:.3f}  best {best_acc:.3f}  "
              f"{epoch_time:.0f}s  peak {peak_gb:.1f} GB  |W_rec| {row['rec_weight_norm']}", flush=True)

        save_atomic({
            "model": net.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_acc": best_acc, "last_acc": last_acc,
        }, ckpt_path)

    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump({"best_test_acc": best_acc, "final_test_acc": last_acc,
                   "params": n_params, "total_params": n_total, "rec_params": n_rec,
                   "lif_neurons_per_step": n_lif, "args": vars(args)}, f, indent=2)
    print(f"Done. Best test accuracy: {best_acc:.4f}   final epoch: {last_acc}")


if __name__ == "__main__":
    main()

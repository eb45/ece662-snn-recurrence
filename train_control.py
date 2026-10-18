"""
Train the feed-forward SNN control on DVS128 Gesture.

Quick debug run (2 epochs) to check shapes, memory, and time per epoch:
    python train_control.py --root /work/eb408/dvs128_gesture --out /work/eb408/ece662_runs/debug --epochs 2

Full run:
    python train_control.py --root /work/eb408/dvs128_gesture --out /work/eb408/ece662_runs/control_s42 --epochs 64

Checkpoints are written every epoch and the script resumes automatically
if <out>/checkpoint.pt exists, so a preempted scavenger job can requeue and
carry on.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional, neuron

from dvs_data import DEFAULT_CONFIG, get_loaders, set_seed
from evaluate import evaluate
from models import CSNN


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
    p.add_argument("--cupy", action="store_true", help="Use the cupy LIF backend (needs cupy installed)")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


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

    model = CSNN(channels=args.channels, num_classes=DEFAULT_CONFIG["num_classes"]).to(device)
    if args.cupy:
        functional.set_backend(model, "cupy", instance=neuron.LIFNode)

    n_params, n_lif = model.num_params(), model.num_lif_neurons()
    print(f"params: {n_params:,}   LIF neurons per step: {n_lif:,}")
    print(f"batch {args.batch_size} x accum {args.accum} = effective {args.batch_size * args.accum}, "
          f"T={args.T}, amp={use_amp}, opt={args.opt}, lr={args.lr}, loss={args.loss}")

    if args.opt == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    ckpt_path = os.path.join(args.out, "checkpoint.pt")
    best_path = os.path.join(args.out, "best.pt")
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    start_epoch, best_acc = 0, 0.0

    if os.path.exists(ckpt_path) and not args.no_resume:
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        start_epoch, best_acc = ck["epoch"] + 1, ck["best_acc"]
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

            scaler.scale(loss / args.accum).backward()
            if (i + 1) % args.accum == 0 or (i + 1) == n_batches:
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
        epoch_time = time.time() - t0
        peak_gb = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0

        if test_acc > best_acc:
            best_acc = test_acc
            save_atomic(model.state_dict(), best_path)

        row = {
            "epoch": epoch, "train_loss": run_loss / run_total,
            "train_acc": run_correct / run_total, "test_acc": test_acc,
            "best_test_acc": best_acc, "train_sec": round(train_time, 1),
            "epoch_sec": round(epoch_time, 1), "peak_gpu_gb": round(peak_gb, 2),
        }
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  "
              f"train {row['train_acc']:.3f}  test {test_acc:.3f}  best {best_acc:.3f}  "
              f"{epoch_time:.0f}s  peak {peak_gb:.1f} GB", flush=True)

        save_atomic({
            "model": model.state_dict(), "opt": opt.state_dict(),
            "sched": sched.state_dict(), "scaler": scaler.state_dict(),
            "epoch": epoch, "best_acc": best_acc,
        }, ckpt_path)

    with open(os.path.join(args.out, "final.json"), "w") as f:
        json.dump({"best_test_acc": best_acc, "params": n_params,
                   "lif_neurons_per_step": n_lif, "args": vars(args)}, f, indent=2)
    print(f"Done. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

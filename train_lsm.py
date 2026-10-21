"""Train the fixed-reservoir LSM on DVS128 Gesture.

Defaults match the feed-forward control's data, optimizer, loss, schedule,
batching, and evaluation settings. Only the linear readout is trained; the
random recurrent liquid remains fixed.
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from dvs_data import DEFAULT_CONFIG, get_loaders, set_seed
from evaluate import evaluate
from lsm_model import BASELINE_LIF_NEURONS, LSM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="DVS128 Gesture root (contains download/)")
    parser.add_argument("--out", required=True, help="Output folder for checkpoints and metrics")
    parser.add_argument("--epochs", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--accum", type=int, default=2)
    parser.add_argument("--T", type=int, default=DEFAULT_CONFIG["T"])
    parser.add_argument("--lif-neurons", type=int, default=BASELINE_LIF_NEURONS,
                        help="Liquid LIF count; defaults to the control model's count")
    parser.add_argument("--readout-features", type=int, default=512)
    parser.add_argument("--recurrent-degree", type=int, default=4)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--input-scale", type=float, default=0.1)
    parser.add_argument("--recurrent-gain", type=float, default=0.25)
    parser.add_argument("--opt", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--loss", choices=["mse", "ce"], default="mse")
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def save_atomic(obj, path):
    temporary_path = path + ".tmp"
    torch.save(obj, temporary_path)
    os.replace(temporary_path, path)


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.accum <= 0:
        raise ValueError("epochs, batch-size, and accum must be positive")

    os.makedirs(args.out, exist_ok=True)
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: no GPU found, training will be very slow")
    use_amp = (not args.no_amp) and device.type == "cuda"

    train_loader, test_loader = get_loaders(
        args.root,
        batch_size=args.batch_size,
        T=args.T,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    model = LSM(
        num_classes=DEFAULT_CONFIG["num_classes"],
        lif_neurons=args.lif_neurons,
        readout_features=args.readout_features,
        tau=args.tau,
        threshold=args.threshold,
        input_scale=args.input_scale,
        recurrent_gain=args.recurrent_gain,
        recurrent_degree=args.recurrent_degree,
        seed=args.seed,
    ).to(device)

    n_params = model.num_params()
    n_lif = model.num_lif_neurons()
    print(f"params: {n_params:,}   LIF neurons per step: {n_lif:,}")
    print(
        f"batch {args.batch_size} x accum {args.accum} = "
        f"effective {args.batch_size * args.accum}, T={args.T}, "
        f"amp={use_amp}, opt={args.opt}, lr={args.lr}, loss={args.loss}"
    )

    if args.opt == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    checkpoint_path = os.path.join(args.out, "checkpoint.pt")
    best_path = os.path.join(args.out, "best.pt")
    metrics_path = os.path.join(args.out, "metrics.jsonl")
    start_epoch, best_acc = 0, 0.0

    if os.path.exists(checkpoint_path) and not args.no_resume:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["opt"])
        scheduler.load_state_dict(checkpoint["sched"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, best_acc = checkpoint["epoch"] + 1, checkpoint["best_acc"]
        print(f"Resumed from epoch {start_epoch} (best test acc so far {best_acc:.4f})")

    num_classes = DEFAULT_CONFIG["num_classes"]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        start_time = time.time()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        running_loss, running_correct, running_total = 0.0, 0, 0
        optimizer.zero_grad(set_to_none=True)
        num_batches = len(train_loader)

        for batch_index, (frames, labels) in enumerate(train_loader):
            frames = frames.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, enabled=use_amp):
                output = model(frames).mean(0)
            output = output.float()
            if args.loss == "mse":
                loss = F.mse_loss(output, F.one_hot(labels, num_classes).float())
            else:
                loss = F.cross_entropy(output, labels)

            scaler.scale(loss / args.accum).backward()
            if (batch_index + 1) % args.accum == 0 or (batch_index + 1) == num_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            functional.reset_net(model)
            running_loss += loss.item() * labels.numel()
            running_correct += (output.argmax(1) == labels).sum().item()
            running_total += labels.numel()

        scheduler.step()
        train_time = time.time() - start_time
        test_acc = evaluate(model, test_loader, device, amp=use_amp)
        epoch_time = time.time() - start_time
        peak_gb = (
            torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
        )

        if test_acc > best_acc:
            best_acc = test_acc
            save_atomic(model.state_dict(), best_path)

        row = {
            "epoch": epoch,
            "train_loss": running_loss / running_total,
            "train_acc": running_correct / running_total,
            "test_acc": test_acc,
            "best_test_acc": best_acc,
            "train_sec": round(train_time, 1),
            "epoch_sec": round(epoch_time, 1),
            "peak_gpu_gb": round(peak_gb, 2),
        }
        with open(metrics_path, "a", encoding="utf-8") as metrics_file:
            metrics_file.write(json.dumps(row) + "\n")
        print(
            f"epoch {epoch + 1}/{args.epochs}  loss {row['train_loss']:.4f}  "
            f"train {row['train_acc']:.3f}  test {test_acc:.3f}  "
            f"best {best_acc:.3f}  {epoch_time:.0f}s  peak {peak_gb:.1f} GB",
            flush=True,
        )

        save_atomic(
            {
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "best_acc": best_acc,
            },
            checkpoint_path,
        )

    with open(os.path.join(args.out, "final.json"), "w", encoding="utf-8") as summary_file:
        json.dump(
            {
                "best_test_acc": best_acc,
                "params": n_params,
                "lif_neurons_per_step": n_lif,
                "args": vars(args),
            },
            summary_file,
            indent=2,
        )
    print(f"Done. Best test accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()

"""
Shared DVS128 Gesture data pipeline for the ECE662 project.

Everyone (feed-forward control, LSM, recurrent SNN) imports this file so all
three arms see the same binned frames, the same split, and the same settings.

Setup:
    pip install torch spikingjelly

    1. Download DvsGesture.tar.gz from IBM's DVS128 Gesture page.
    2. Put it at <ROOT>/download/DvsGesture.tar.gz
       (SpikingJelly does not download it for you.)
    3. Run this file once to build the cache:
           python dvs_data.py --root /path/to/DVS128Gesture
       The first run extracts the archive and converts events to binned
       .npz frames under <ROOT>/frames_number_<T>_split_by_number/.
       This takes a while and only happens once. Teammates pointing at the
       same ROOT reuse the cache.

Usage in training code:
    from dvs_data import get_loaders, DEFAULT_CONFIG

    train_loader, test_loader = get_loaders(root, batch_size=16)
    for frames, labels in train_loader:
        # frames: [T, B, 2, 128, 128], labels: [B]
        ...
"""

import argparse
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from spikingjelly.datasets.dvs128_gesture import DVS128Gesture

# Shared settings. Change these in one place so every arm stays matched.
DEFAULT_CONFIG = {
    "T": 16,               # number of time bins per clip
    "split_by": "number",  # bins hold equal numbers of events
    "batch_size": 16,
    "num_workers": 4,
    "seed": 2026,
    "num_classes": 11,
}


def set_seed(seed: int = DEFAULT_CONFIG["seed"]) -> None:
    """Seed python, numpy, and torch so runs are comparable across arms."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init_fn(worker_id: int) -> None:
    # Make each dataloader worker's numpy/random state deterministic.
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


def get_datasets(root: str, T: int = DEFAULT_CONFIG["T"],
                 split_by: str = DEFAULT_CONFIG["split_by"]):
    """Return (train_set, test_set) of binned frames.

    The standard DVS128 Gesture split is by person, so the test set is
    made of unseen people. Each item is (frames, label) with frames shaped
    [T, 2, 128, 128] (channel 0/1 = ON/OFF event counts per pixel).
    """
    common = dict(root=root, data_type="frame", frames_number=T, split_by=split_by)
    train_set = DVS128Gesture(train=True, **common)
    test_set = DVS128Gesture(train=False, **common)
    return train_set, test_set


class _TimeFirst:
    """Collate wrapper that returns frames as [T, B, 2, 128, 128].

    SpikingJelly's multi-step mode expects time first, so this saves each
    arm from remembering to transpose.
    """

    def __call__(self, batch):
        frames = torch.stack([torch.as_tensor(b[0]) for b in batch], dim=0)  # [B,T,2,H,W]
        labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
        return frames.transpose(0, 1).contiguous().float(), labels


def get_loaders(root: str,
                batch_size: int = DEFAULT_CONFIG["batch_size"],
                T: int = DEFAULT_CONFIG["T"],
                num_workers: int = DEFAULT_CONFIG["num_workers"],
                seed: int = DEFAULT_CONFIG["seed"],
                split_by: str = DEFAULT_CONFIG["split_by"]):
    """Return (train_loader, test_loader). Batches are ([T,B,2,128,128], [B])."""
    set_seed(seed)
    train_set, test_set = get_datasets(root, T=T, split_by=split_by)

    g = torch.Generator()
    g.manual_seed(seed)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=pin, collate_fn=_TimeFirst(),
        worker_init_fn=_worker_init_fn, generator=g,
        persistent_workers=num_workers > 0,
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False,
        num_workers=num_workers, pin_memory=pin, collate_fn=_TimeFirst(),
        worker_init_fn=_worker_init_fn,
        persistent_workers=num_workers > 0,
    )
    return train_loader, test_loader


def sanity_check(root: str, T: int = DEFAULT_CONFIG["T"], batch_size: int = 4) -> None:
    """Build the cache (first run) and check shapes, labels, and split sizes."""
    train_set, test_set = get_datasets(root, T=T)
    print(f"train clips: {len(train_set)}   test clips: {len(test_set)}")

    train_loader, _ = get_loaders(root, batch_size=batch_size, T=T, num_workers=0)
    frames, labels = next(iter(train_loader))
    print(f"batch frames: {tuple(frames.shape)}  (expect [T, B, 2, 128, 128])")
    print(f"batch labels: {labels.tolist()}  (expect ints in 0..10)")
    print(f"frame dtype: {frames.dtype}   min: {frames.min().item():.1f}   "
          f"max: {frames.max().item():.1f}   mean: {frames.mean().item():.4f}")

    assert frames.shape == (T, batch_size, 2, 128, 128), "unexpected frame shape"
    assert labels.min() >= 0 and labels.max() < DEFAULT_CONFIG["num_classes"], "bad labels"
    print("OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the DVS128 Gesture frame cache and sanity check it.")
    parser.add_argument("--root", required=True,
                        help="Dataset root containing download/DvsGesture.tar.gz")
    parser.add_argument("--T", type=int, default=DEFAULT_CONFIG["T"])
    args = parser.parse_args()
    sanity_check(args.root, T=args.T)
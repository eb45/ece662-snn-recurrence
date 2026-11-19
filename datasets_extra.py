"""
Extra datasets for the dataset-scouting branch, tested with the feed-forward
control only. DVS128 Gesture stays in dvs_data.py and is unchanged.

Every loader here returns the same time-first batches as dvs_data.get_loaders:
frames [T, B, ...], labels [B].

DVS-Gesture-Chain (DVS-GC)
--------------------------
Chains of seq_len DVS128 Gesture clips, built by the dvsgc package from the
DVS128 Gesture events we already have. With seq_len=4, class_num=3 there are
3**4 = 81 chain classes, so a model that ignores order cannot solve it.

Install on DCC with --no-deps. dvsgc pins spikingjelly==0.0.0.0.8, which would
replace our SpikingJelly 2.0.0; the functions it calls exist in 2.0.0 too.
    pip install --no-deps dvsgc

Notes on dvsgc behavior that shape this loader:
  * dvsgc names its cache only by T and split_by, so different seq_len or
    class_num settings would collide. Each config gets its own cache folder
    (see dvsgc_cache_root).
  * dvsg_path must be the events_np folder that SpikingJelly made when it
    built DVS128 Gesture. If it is missing, dvsgc falls back to extracting
    the raw archive into the cache folder, so we refuse to start instead.
  * dvsgc carves a validation split out of train (validation=0.2 by
    default). We train on "train" and score on "test", like DVS128.

Build the cache and sanity check (run on DCC, not the laptop):
    python datasets_extra.py --dataset dvsgc --root /work/eb408/dvs128_gesture
"""

import argparse
import os
from collections import Counter

import torch

from dvs_data import DEFAULT_CONFIG, check_batch, get_loaders, make_loaders, set_seed

DVSGC_DEFAULTS = {
    "T": 16,
    "seq_len": 4,
    "class_num": 3,
    "validation": 0.2,
    "split_by": "number",
    "alpha_min": 0.5,
    "alpha_max": 0.7,
    "repeat": True,
}

# Static facts per dataset. num_classes for dvsgc depends on its config.
DATASETS = {
    "dvs128": {"frame_shape": (2, 128, 128), "num_classes": DEFAULT_CONFIG["num_classes"]},
    "dvsgc": {"frame_shape": (2, 128, 128), "num_classes": None},
}


def dvsgc_num_classes(seq_len: int = DVSGC_DEFAULTS["seq_len"],
                      class_num: int = DVSGC_DEFAULTS["class_num"],
                      repeat: bool = DVSGC_DEFAULTS["repeat"]) -> int:
    """Number of chain classes dvsgc creates for this config."""
    if repeat:
        return class_num ** seq_len
    return class_num * (class_num - 1) ** (seq_len - 1)


def dvsgc_cache_root(dvs128_root: str,
                     seq_len: int = DVSGC_DEFAULTS["seq_len"],
                     class_num: int = DVSGC_DEFAULTS["class_num"],
                     validation: float = DVSGC_DEFAULTS["validation"],
                     alpha_min: float = DVSGC_DEFAULTS["alpha_min"],
                     alpha_max: float = DVSGC_DEFAULTS["alpha_max"],
                     repeat: bool = DVSGC_DEFAULTS["repeat"]) -> str:
    """One cache folder per DVS-GC config, under <dvs128_root>/dvsgc/.

    dvsgc adds its own DVSGC_frames_number_<T>_split_by_<split_by> folder
    inside this, so T and split_by are covered there.
    """
    name = (f"seq{seq_len}_cls{class_num}_val{validation}"
            f"_a{alpha_min}-{alpha_max}" + ("" if repeat else "_norepeat"))
    return os.path.join(dvs128_root, "dvsgc", name)


def get_dvsgc_datasets(dvs128_root: str,
                       T: int = DVSGC_DEFAULTS["T"],
                       cache_root: str = None,
                       seq_len: int = DVSGC_DEFAULTS["seq_len"],
                       class_num: int = DVSGC_DEFAULTS["class_num"],
                       validation: float = DVSGC_DEFAULTS["validation"],
                       split_by: str = DVSGC_DEFAULTS["split_by"],
                       alpha_min: float = DVSGC_DEFAULTS["alpha_min"],
                       alpha_max: float = DVSGC_DEFAULTS["alpha_max"],
                       repeat: bool = DVSGC_DEFAULTS["repeat"]):
    """Return (train_set, test_set). Items are (frames [T, 2, 128, 128], label).

    The first call builds the cache, which takes a while. Later calls reuse it.
    """
    import dvsgc  # imported here so the laptop tests do not need it

    events_np = os.path.join(dvs128_root, "events_np")
    if not os.path.isdir(events_np):
        raise FileNotFoundError(
            f"{events_np} not found. Build the DVS128 Gesture cache first "
            f"(python dvs_data.py --root {dvs128_root}).")

    if cache_root is None:
        cache_root = dvsgc_cache_root(dvs128_root, seq_len, class_num, validation,
                                      alpha_min, alpha_max, repeat)
    os.makedirs(cache_root, exist_ok=True)

    common = dict(root=cache_root, frames_number=T, validation=validation,
                  split_by=split_by, alpha_min=alpha_min, alpha_max=alpha_max,
                  seq_len=seq_len, class_num=class_num, repeat=repeat,
                  dvsg_path=events_np)
    train_set = dvsgc.DVSGestureChain(split="train", **common)
    test_set = dvsgc.DVSGestureChain(split="test", **common)

    expected = dvsgc_num_classes(seq_len, class_num, repeat)
    for name, ds in (("train", train_set), ("test", test_set)):
        if len(ds.classes) != expected:
            raise RuntimeError(f"DVS-GC {name} has {len(ds.classes)} classes, expected {expected}. "
                               f"The cache at {cache_root} may be from another config.")
    return train_set, test_set


def get_dataset_loaders(dataset: str, root: str,
                        batch_size: int = DEFAULT_CONFIG["batch_size"],
                        T: int = DEFAULT_CONFIG["T"],
                        num_workers: int = DEFAULT_CONFIG["num_workers"],
                        seed: int = DEFAULT_CONFIG["seed"],
                        **dataset_kwargs):
    """Return (train_loader, test_loader, info) for any supported dataset.

    info has num_classes, frame_shape, and cache_root (None for dvs128).
    dvs128 goes through dvs_data.get_loaders unchanged.
    """
    if dataset == "dvs128":
        train_loader, test_loader = get_loaders(root, batch_size=batch_size, T=T,
                                                num_workers=num_workers, seed=seed)
        info = dict(DATASETS["dvs128"], cache_root=None)
        return train_loader, test_loader, info

    if dataset == "dvsgc":
        cfg = {k: v for k, v in DVSGC_DEFAULTS.items() if k != "T"}
        cfg.update(dataset_kwargs)
        cache_root = cfg.pop("cache_root", None) or dvsgc_cache_root(
            root, cfg["seq_len"], cfg["class_num"], cfg["validation"],
            cfg["alpha_min"], cfg["alpha_max"], cfg["repeat"])
        set_seed(seed)
        train_set, test_set = get_dvsgc_datasets(root, T=T, cache_root=cache_root, **cfg)
        # dvsgc reseeds python's random module on import, so seed again.
        set_seed(seed)
        train_loader, test_loader = make_loaders(train_set, test_set, batch_size=batch_size,
                                                 num_workers=num_workers, seed=seed)
        info = dict(DATASETS["dvsgc"], cache_root=cache_root,
                    num_classes=dvsgc_num_classes(cfg["seq_len"], cfg["class_num"], cfg["repeat"]))
        return train_loader, test_loader, info

    raise ValueError(f"unknown dataset {dataset!r}, choose from {sorted(DATASETS)}")


class TimeShuffle:
    """Randomly permute the time bins of a time-first batch [T, B, ...].

    Each sample gets its own permutation. Use with evaluate's transform hook:
        evaluate(model, test_loader, device, transform=TimeShuffle(seed=0))
    If accuracy barely drops, the task does not need temporal order.
    The seed makes the shuffle reproducible across runs.
    """

    def __init__(self, seed: int = 0):
        self.gen = torch.Generator().manual_seed(seed)

    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        T, B = frames.shape[:2]
        perm = torch.rand(T, B, generator=self.gen).argsort(dim=0).to(frames.device)  # [T, B]
        idx = perm.view(T, B, *([1] * (frames.dim() - 2))).expand_as(frames)
        return frames.gather(0, idx)


def _warn_dvsgc_rebuild(split: str, counts: Counter, validation) -> None:
    # dvsgc rebuilds every class folder with fewer files than these thresholds
    # each time the dataset is constructed, with new random chain lengths.
    # If that happens the cache is not stable between runs.
    threshold = (int(97 * (1 - validation)) if validation is not None else 97) if split == "train" else 24
    short = sum(1 for c in counts.values() if c < threshold)
    if short:
        print(f"WARNING: {short} {split} classes have fewer than {threshold} clips. dvsgc will "
              f"rebuild them on every load, so the cache changes between runs.")


def _dir_size_gb(path: str) -> float:
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(dirpath, f))
    return total / 1e9


def sanity_check(dataset: str, root: str, T: int, batch_size: int = 4, **dataset_kwargs) -> None:
    """Build the cache (first run) and check sizes, shapes, labels, and class balance."""
    train_loader, test_loader, info = get_dataset_loaders(
        dataset, root, batch_size=batch_size, T=T, num_workers=0, **dataset_kwargs)
    n_cls = info["num_classes"]
    train_set, test_set = train_loader.dataset, test_loader.dataset
    print(f"dataset: {dataset}   T: {T}   classes: {n_cls}   chance: {1 / n_cls:.4f}")
    print(f"train clips: {len(train_set)}   test clips: {len(test_set)}")

    for name, ds in (("train", train_set), ("test", test_set)):
        if hasattr(ds, "targets"):
            counts = Counter(ds.targets)
            print(f"{name} per-class clips: min {min(counts.values())}  max {max(counts.values())}  "
                  f"classes present {len(counts)}/{n_cls}")
            assert len(counts) == n_cls, f"{name} is missing classes"
            if dataset == "dvsgc":
                _warn_dvsgc_rebuild(name, counts, dataset_kwargs.get("validation", DVSGC_DEFAULTS["validation"]))

    frames, labels = next(iter(train_loader))
    print(f"batch frames: {tuple(frames.shape)}  (expect [T, B, {', '.join(map(str, info['frame_shape']))}])")
    print(f"batch labels: {labels.tolist()}  (expect ints in 0..{n_cls - 1})")
    print(f"frame dtype: {frames.dtype}   min: {frames.min().item():.1f}   "
          f"max: {frames.max().item():.1f}   mean: {frames.mean().item():.4f}")
    nonzero_bins = (frames.flatten(2).sum(-1) > 0).float().mean().item()
    print(f"fraction of (t, sample) bins with events: {nonzero_bins:.3f}")
    check_batch(frames, labels, T=T, batch_size=batch_size,
                num_classes=n_cls, frame_shape=info["frame_shape"])

    if info["cache_root"] is not None:
        print(f"cache: {info['cache_root']}   size: {_dir_size_gb(info['cache_root']):.2f} GB")
    print("OK")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build an extra dataset's cache and sanity check it.")
    p.add_argument("--dataset", choices=sorted(DATASETS), required=True)
    p.add_argument("--root", required=True,
                   help="DVS128 Gesture root. For dvsgc the cache goes in <root>/dvsgc/<config>.")
    p.add_argument("--T", type=int, default=DEFAULT_CONFIG["T"])
    p.add_argument("--seq-len", type=int, default=DVSGC_DEFAULTS["seq_len"])
    p.add_argument("--class-num", type=int, default=DVSGC_DEFAULTS["class_num"])
    args = p.parse_args()

    kwargs = {}
    if args.dataset == "dvsgc":
        kwargs = dict(seq_len=args.seq_len, class_num=args.class_num)
    sanity_check(args.dataset, args.root, T=args.T, **kwargs)

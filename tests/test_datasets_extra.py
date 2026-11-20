"""CPU checks for datasets_extra.py and the --dataset flag in train_control.py.

    python -m pytest tests/test_datasets_extra.py -q

No downloads. dvsgc is replaced by a fake module, except in the one test
that builds a tiny real DVS-GC cache from synthetic events; that test is
skipped if dvsgc is not installed (pip install --no-deps dvsgc).
"""

import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import Dataset

import datasets_extra
import dvs_data
from datasets_extra import (TimeShuffle, dvsgc_cache_root, dvsgc_num_classes,
                            get_dataset_loaders)
from evaluate import evaluate


class _FakeClips(Dataset):
    """(frames [T, 2, H, W] float32, label) items, like the SpikingJelly folders."""

    def __init__(self, n, T, num_classes, size=128, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.frames = torch.randint(0, 3, (n, T, 2, size, size), generator=g).float().numpy()
        self.targets = [i % num_classes for i in range(n)]
        self.classes = [str(i) for i in range(num_classes)]

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.frames[i], self.targets[i]


def _fake_dvsgc_module(calls):
    mod = types.ModuleType("dvsgc")

    class DVSGestureChain(_FakeClips):
        def __init__(self, root, frames_number, split, seq_len, class_num, repeat, **kw):
            calls.append(dict(root=root, frames_number=frames_number, split=split,
                              seq_len=seq_len, class_num=class_num, repeat=repeat, **kw))
            n_cls = dvsgc_num_classes(seq_len, class_num, repeat)
            super().__init__(n=n_cls if split == "test" else 2 * n_cls,
                             T=frames_number, num_classes=n_cls, size=8)

    mod.DVSGestureChain = DVSGestureChain
    return mod


class TimeShuffleTests(unittest.TestCase):
    def test_permutes_time_per_sample(self):
        x = torch.arange(6 * 3).float().view(6, 3, 1, 1, 1).expand(6, 3, 2, 4, 4).contiguous()
        y = TimeShuffle(seed=0)(x)
        self.assertEqual(y.shape, x.shape)
        for b in range(3):
            # Same bins, new order, each bin moved as a whole.
            self.assertTrue(torch.equal(y[:, b].sort(dim=0).values, x[:, b]))
            self.assertTrue((y[:, b] == y[:, b, :1, :1, :1]).all())
        self.assertFalse(torch.equal(y, x))
        orders = {tuple(y[:, b, 0, 0, 0].tolist()) for b in range(3)}
        self.assertGreater(len(orders), 1, "samples should get different permutations")

    def test_reproducible_with_seed(self):
        x = torch.randn(8, 4, 5)
        self.assertTrue(torch.equal(TimeShuffle(seed=3)(x), TimeShuffle(seed=3)(x)))
        self.assertFalse(torch.equal(TimeShuffle(seed=3)(x), TimeShuffle(seed=4)(x)))

    def test_works_as_evaluate_transform(self):
        # A "model" that predicts the class from the first time bin only, so
        # shuffling time must hurt it.
        T, B = 8, 16

        class FirstBin(torch.nn.Module):
            def forward(self, x):
                out = torch.zeros(x.shape[0], x.shape[1], T)
                out[:, torch.arange(x.shape[1]), x[0, :, 0].long()] = 1
                return out

        labels = torch.arange(B) % T
        frames = torch.zeros(T, B, 1)
        frames[0, :, 0] = labels.float()
        frames[1:, :, 0] = (labels.float() + 1) % T
        loader = [(frames, labels)]
        dev = torch.device("cpu")
        with patch("evaluate.functional.reset_net"):
            clean = evaluate(FirstBin(), loader, dev)
            shuffled = evaluate(FirstBin(), loader, dev, transform=TimeShuffle(seed=0))
        self.assertEqual(clean, 1.0)
        self.assertLess(shuffled, 0.5)


class SharedDefaultsTests(unittest.TestCase):
    def test_dvs128_defaults_unchanged(self):
        cfg = dvs_data.DEFAULT_CONFIG
        self.assertEqual((cfg["T"], cfg["split_by"], cfg["num_classes"], cfg["frame_shape"]),
                         (16, "number", 11, (2, 128, 128)))

    def test_check_batch(self):
        f, l = torch.zeros(16, 4, 2, 128, 128), torch.tensor([0, 3, 10, 1])
        dvs_data.check_batch(f, l, T=16, batch_size=4)
        with self.assertRaises(AssertionError):
            dvs_data.check_batch(f, torch.tensor([0, 1, 2, 11]), T=16, batch_size=4)
        with self.assertRaises(AssertionError):
            dvs_data.check_batch(torch.zeros(16, 4, 2, 64, 64), l, T=16, batch_size=4)
        dvs_data.check_batch(torch.zeros(4, 2, 700), torch.tensor([19, 0]), T=4, batch_size=2,
                             num_classes=20, frame_shape=(700,))

    def test_make_loaders_time_first(self):
        tr, te = _FakeClips(6, T=5, num_classes=3, size=8), _FakeClips(3, T=5, num_classes=3, size=8)
        train_loader, test_loader = dvs_data.make_loaders(tr, te, batch_size=2, num_workers=0)
        frames, labels = next(iter(train_loader))
        self.assertEqual(tuple(frames.shape), (5, 2, 2, 8, 8))
        self.assertEqual(frames.dtype, torch.float32)
        self.assertEqual(tuple(labels.shape), (2,))
        self.assertEqual(len(test_loader.dataset), 3)

    def test_dvs128_goes_through_get_loaders(self):
        with patch("datasets_extra.get_loaders", return_value=("tr", "te")) as gl:
            tr, te, info = get_dataset_loaders("dvs128", "/data/dvs", batch_size=8, T=16,
                                               num_workers=2, seed=5)
        gl.assert_called_once_with("/data/dvs", batch_size=8, T=16, num_workers=2, seed=5)
        self.assertEqual((tr, te), ("tr", "te"))
        self.assertEqual(info["num_classes"], 11)
        self.assertEqual(info["frame_shape"], (2, 128, 128))

    def test_unknown_dataset(self):
        with self.assertRaises(ValueError):
            get_dataset_loaders("nope", "/tmp")


class DVSGCLoaderTests(unittest.TestCase):
    def test_num_classes(self):
        self.assertEqual(dvsgc_num_classes(4, 3), 81)
        self.assertEqual(dvsgc_num_classes(4, 3, repeat=False), 24)

    def test_cache_root_is_per_config(self):
        a = dvsgc_cache_root("/r", seq_len=4, class_num=3)
        self.assertTrue(a.startswith(os.path.join("/r", "dvsgc")))
        self.assertNotEqual(a, dvsgc_cache_root("/r", seq_len=4, class_num=4))
        self.assertNotEqual(a, dvsgc_cache_root("/r", seq_len=3, class_num=3))
        self.assertNotEqual(a, dvsgc_cache_root("/r", seq_len=4, class_num=3, validation=None))

    def test_loader_with_fake_dvsgc(self):
        calls = []
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(sys.modules, {"dvsgc": _fake_dvsgc_module(calls)}):
            os.makedirs(os.path.join(root, "events_np"))
            train_loader, test_loader, info = get_dataset_loaders(
                "dvsgc", root, batch_size=4, T=16, num_workers=0)
            frames, labels = next(iter(train_loader))

            self.assertEqual(info["num_classes"], 81)
            self.assertEqual(tuple(frames.shape), (16, 4, 2, 8, 8))
            self.assertTrue(((labels >= 0) & (labels < 81)).all())
            self.assertEqual([c["split"] for c in calls], ["train", "test"])
            for c in calls:
                self.assertEqual(c["dvsg_path"], os.path.join(root, "events_np"))
                self.assertEqual(c["root"], dvsgc_cache_root(root))
                self.assertEqual((c["frames_number"], c["seq_len"], c["class_num"]), (16, 4, 3))
                self.assertEqual(c["split_by"], "number")
            self.assertTrue(os.path.isdir(info["cache_root"]))

    def test_missing_events_np_fails_fast(self):
        with tempfile.TemporaryDirectory() as root, \
                patch.dict(sys.modules, {"dvsgc": _fake_dvsgc_module([])}):
            with self.assertRaises(FileNotFoundError):
                get_dataset_loaders("dvsgc", root, batch_size=2, num_workers=0)
            self.assertFalse(os.path.exists(os.path.join(root, "dvsgc")))


class RealDVSGCBuildTest(unittest.TestCase):
    """Builds a tiny DVS-GC cache with the real dvsgc package from synthetic events."""

    def test_build_from_synthetic_events(self):
        try:
            import dvsgc  # noqa: F401
        except ImportError:
            self.skipTest("dvsgc not installed")

        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as root:
            # Same layout SpikingJelly 2.0.0 writes: events_np/<split>/<label>/<name>.npz
            for split, users in (("train", range(1, 6)), ("test", range(6, 8))):
                for label in range(11):
                    d = os.path.join(root, "events_np", split, str(label))
                    os.makedirs(d)
                    for u in users:
                        n = 300
                        np.savez(os.path.join(d, f"user{u:02d}_led_0.npz"),
                                 t=np.sort(rng.integers(0, 10**6, n)).astype(np.int64),
                                 x=rng.integers(0, 128, n).astype(np.uint16),
                                 y=rng.integers(0, 128, n).astype(np.uint16),
                                 p=rng.integers(0, 2, n).astype(np.uint8))

            T = 8
            with patch("builtins.print"):
                train_loader, test_loader, info = get_dataset_loaders(
                    "dvsgc", root, batch_size=2, T=T, num_workers=0, seq_len=2, class_num=2)
            self.assertEqual(info["num_classes"], 4)
            self.assertEqual(len(train_loader.dataset.classes), 4)
            self.assertEqual(len(test_loader.dataset), 4 * 2)
            frames, labels = next(iter(test_loader))
            self.assertEqual(tuple(frames.shape), (T, 2, 2, 128, 128))
            self.assertGreater(frames.sum().item(), 0)


class TrainControlDatasetFlagTests(unittest.TestCase):
    def _run(self, dataset, num_classes, extra=()):
        import train_control

        tr = _FakeClips(4, T=2, num_classes=num_classes)
        te = _FakeClips(2, T=2, num_classes=num_classes, seed=1)
        loaders = dvs_data.make_loaders(tr, te, batch_size=2, num_workers=0)
        info = dict(num_classes=num_classes, frame_shape=(2, 128, 128), cache_root=None)
        with tempfile.TemporaryDirectory() as out:
            argv = ["train_control.py", "--dataset", dataset, "--root", "/unused", "--out", out,
                    "--epochs", "1", "--batch-size", "2", "--accum", "1", "--T", "2",
                    "--channels", "4", "--num-workers", "0", *extra]
            with patch.object(sys, "argv", argv), \
                    patch("train_control.get_dataset_loaders",
                          return_value=(*loaders, info)) as gdl, \
                    patch("builtins.print"):
                train_control.main()
            with open(os.path.join(out, "final.json")) as f:
                final = json.load(f)
        return gdl, final

    def test_default_is_dvs128_with_11_classes(self):
        import train_control
        with patch.object(sys, "argv", ["x", "--root", "r", "--out", "o"]):
            self.assertEqual(train_control.parse_args().dataset, "dvs128")
        gdl, final = self._run("dvs128", 11)
        self.assertEqual(gdl.call_args.args[0], "dvs128")
        self.assertNotIn("seq_len", gdl.call_args.kwargs)
        self.assertEqual(final["num_classes"], 11)

    def test_dvsgc_passes_num_classes_through(self):
        gdl, final = self._run("dvsgc", 81)
        self.assertEqual(gdl.call_args.args[0], "dvsgc")
        self.assertEqual((gdl.call_args.kwargs["seq_len"], gdl.call_args.kwargs["class_num"]), (4, 3))
        self.assertEqual(final["num_classes"], 81)
        self.assertEqual(final["dataset"], "dvsgc")


class EvalShuffleTests(unittest.TestCase):
    def test_end_to_end_on_tiny_run(self):
        import eval_shuffle
        import train_control

        tr = _FakeClips(4, T=3, num_classes=81)
        te = _FakeClips(4, T=3, num_classes=81, seed=1)
        info = dict(num_classes=81, frame_shape=(2, 128, 128), cache_root=None)

        def fake_loaders(*a, **kw):
            return (*dvs_data.make_loaders(tr, te, batch_size=2, num_workers=0), info)

        with tempfile.TemporaryDirectory() as out:
            argv = ["train_control.py", "--dataset", "dvsgc", "--root", "/data", "--out", out,
                    "--epochs", "1", "--batch-size", "2", "--accum", "1", "--T", "3",
                    "--channels", "4", "--num-workers", "0"]
            with patch.object(sys, "argv", argv), \
                    patch("train_control.get_dataset_loaders", side_effect=fake_loaders), \
                    patch("builtins.print"):
                train_control.main()

            with patch("eval_shuffle.get_dataset_loaders", side_effect=fake_loaders) as gdl, \
                    patch("builtins.print"):
                result = eval_shuffle.main(["--run", out, "--seeds", "0", "1", "--num-workers", "0"])

            self.assertEqual(gdl.call_args.args[:2], ("dvsgc", "/data"))
            self.assertEqual(gdl.call_args.kwargs["T"], 3)
            self.assertEqual((gdl.call_args.kwargs["seq_len"], gdl.call_args.kwargs["class_num"]), (4, 3))
            with open(os.path.join(out, "shuffle.json")) as f:
                saved = json.load(f)
        self.assertEqual(saved, json.loads(json.dumps(result)))
        self.assertEqual(len(saved["shuffled_acc"]), 2)
        self.assertAlmostEqual(saved["chance"], 1 / 81)
        self.assertAlmostEqual(saved["drop_abs"], saved["clean_acc"] - saved["shuffled_mean"])
        self.assertTrue(0.0 <= saved["clean_acc"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

"""Checks for the recurrent SNN arm. Run from the repo root:

    python -m unittest tests.test_recurrent -v

Uses a narrow network (4 channels) on CPU, so it runs in well under a minute
and does not need the dataset.
"""

import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader, TensorDataset

import train_recurrent
from models import CSNN
from probe_recurrent import probe
from rec_drift import (backfill, conv_spectral_norm, drift_stats, net_drift_stats,
                       rebuild_w0, rec_weights)
from recurrent_model import LateReadout, RecurrentCSNN, RecurrentLIF, load_run

SMALL = dict(channels=4, hidden=16)


def _frames(T=3, B=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, 6, (T, B, 2, 128, 128), generator=g).float()


def _time_first_collate(batch):
    frames = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return frames.transpose(0, 1).contiguous(), labels


class RecurrentModelTests(unittest.TestCase):
    def test_output_shape_for_any_T(self):
        model = RecurrentCSNN(**SMALL).eval()
        with torch.no_grad():
            for T in (3, 7):  # gapped clips are longer than 16
                out = model(_frames(T=T))
                functional.reset_net(model)
                self.assertEqual(tuple(out.shape), (T, 2, 11))
                self.assertTrue(torch.isfinite(out).all())

    def test_zero_recurrence_matches_control(self):
        # With W_rec = 0 the recurrent net must compute exactly what CSNN does.
        # This checks the per-step loop, the layer swap, and the step modes.
        # Train mode (BN uses batch stats) with dropout off: in eval mode a
        # freshly initialized 4-channel net goes silent by block 5, which
        # would make the comparison vacuous.
        torch.manual_seed(0)
        ctrl = CSNN(dropout=0.0, **SMALL).train()
        rnet = RecurrentCSNN(rec_layers=("conv1", "conv5", "fc"), rec_init="zero",
                             dropout=0.0, **SMALL).train()
        missing, unexpected = rnet.load_state_dict(ctrl.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(missing and all(k.endswith("rec.weight") for k in missing))

        x = _frames(T=4, B=4)
        with torch.no_grad():
            feats_c = ctrl.conv_stack(x)
            functional.reset_net(ctrl)
            feats_r = rnet.conv_stack(x)
            functional.reset_net(rnet)
            self.assertGreater(feats_c.sum().item(), 0, "no spikes; test would be vacuous")
            torch.testing.assert_close(feats_r, feats_c)
            torch.testing.assert_close(rnet(x), ctrl(x))

    def test_recurrent_weights_get_gradients(self):
        torch.manual_seed(1)
        model = RecurrentCSNN(**SMALL).train()
        out = model(_frames(T=4)).mean(0)
        loss = torch.nn.functional.mse_loss(out, torch.nn.functional.one_hot(
            torch.tensor([1, 4]), 11).float())
        loss.backward()
        norms = []
        for name, m in model.rec_modules().items():
            self.assertIsNotNone(m.rec.weight.grad, name)
            self.assertTrue(torch.isfinite(m.rec.weight.grad).all(), name)
            norms.append(m.rec.weight.grad.norm().item())
        self.assertGreater(sum(norms), 0.0)

    def test_freeze_rec(self):
        model = RecurrentCSNN(freeze_rec=True, **SMALL)
        for m in model.rec_modules().values():
            self.assertFalse(m.rec.weight.requires_grad)
            self.assertGreater(m.rec.weight.abs().sum().item(), 0)  # random, not zero
        self.assertEqual(model.num_total_params() - model.num_params(), model.num_rec_params())

        model(_frames()).mean().backward()
        for m in model.rec_modules().values():
            self.assertIsNone(m.rec.weight.grad)

    def test_rejects_bad_configs(self):
        with self.assertRaises(ValueError):
            RecurrentCSNN(rec_layers=("conv6",), **SMALL)
        with self.assertRaises(ValueError):
            RecurrentCSNN(rec_layers=(), **SMALL)
        with self.assertRaises(ValueError):
            RecurrentCSNN(rec_init="zero", freeze_rec=True, **SMALL)

    def test_survives_set_step_mode(self):
        # Other code may call set_step_mode(model, "m") on the whole net.
        model = RecurrentCSNN(**SMALL).eval()
        functional.set_step_mode(model, "m")
        with torch.no_grad():
            out = model(_frames())
        self.assertEqual(tuple(out.shape), (3, 2, 11))
        for m in model.modules():
            if isinstance(m, RecurrentLIF):
                self.assertEqual(m.lif.step_mode, "s")

    def test_reset_makes_forward_repeatable(self):
        model = RecurrentCSNN(**SMALL).eval()
        x = _frames()
        with torch.no_grad():
            first = model(x)
            functional.reset_net(model)
            second = model(x)
        torch.testing.assert_close(first, second)

    def test_late_readout(self):
        net = RecurrentCSNN(**SMALL).eval()
        x = _frames(T=5)
        with torch.no_grad():
            full = net(x)
            functional.reset_net(net)
            late = LateReadout(net, 2)(x)
        torch.testing.assert_close(late, full[-2:])

    def test_full_size_counts_match_control(self):
        ctrl, rnet = CSNN(), RecurrentCSNN(rec_layers=("conv5", "fc"))
        self.assertEqual(rnet.num_lif_neurons(), ctrl.num_lif_neurons())
        self.assertEqual(rnet.num_lif_neurons(), 2_794_094)
        self.assertEqual(rnet.num_rec_params(), 128 * 128 * 9 + 512 * 512)
        self.assertEqual(rnet.num_params(), ctrl.num_params() + rnet.num_rec_params())


class RecurrentTrainingTests(unittest.TestCase):
    def test_metrics_checkpoint_resume_and_load(self):
        frames = _frames(T=2, B=4)
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        loader = DataLoader(TensorDataset(frames.transpose(0, 1), labels), batch_size=2,
                            shuffle=False, collate_fn=_time_first_collate)

        def make_args(output_dir, epochs, readout, rec_lr, rec_wd, anchor):
            return Namespace(
                root="unused", out=output_dir, epochs=epochs, batch_size=2, accum=1, T=2,
                channels=4, opt="adam", lr=1e-3, loss="mse", seed=17, num_workers=0,
                no_amp=True, no_resume=False, rec_layers=["conv5", "fc"],
                rec_init="orthogonal", rec_gain=0.5, freeze_rec=False,
                readout=readout, late_k=1, clip_grad=1.0, rec_lr=rec_lr, rec_wd=rec_wd,
                rec_anchor=anchor,
            )

        # Later configs also exercise the separate W_rec optimizer group (and the
        # anchored decay) across a resume.
        configs = (("mean", None, 0.0, False), ("late", 1e-4, 0.5, False), ("mean", None, 0.5, True))
        for readout, rec_lr, rec_wd, anchor in configs:
            with self.subTest(readout=readout, rec_wd=rec_wd, anchor=anchor), \
                    tempfile.TemporaryDirectory() as out_dir:
                cfg = (readout, rec_lr, rec_wd, anchor)
                with patch.object(train_recurrent, "get_loaders", return_value=(loader, loader)):
                    with patch.object(train_recurrent, "parse_args",
                                      return_value=make_args(out_dir, 1, *cfg)):
                        train_recurrent.main()
                    w0_after_first = torch.load(os.path.join(out_dir, "w0.pt"))
                    with patch.object(train_recurrent, "parse_args",
                                      return_value=make_args(out_dir, 2, *cfg)):
                        train_recurrent.main()

                with open(os.path.join(out_dir, "metrics.jsonl")) as f:
                    rows = [json.loads(line) for line in f]
                self.assertEqual([r["epoch"] for r in rows], [0, 1])
                self.assertEqual(set(rows[0]["rec_weight_norm"]), {"conv5", "fc"})
                self.assertEqual(set(rows[0]["rec_drift"]), {"conv5", "fc", "all"})
                # Whether a 4-channel toy net actually moves W_rec is checked on the
                # cluster debug run; here only that the field is filled in.
                self.assertGreaterEqual(rows[-1]["rec_drift"]["all"]["drift_rel"], 0.0)

                with open(os.path.join(out_dir, "final.json")) as f:
                    final = json.load(f)
                self.assertIsNotNone(final["final_test_acc"])
                self.assertAlmostEqual(final["rec_init_stats"]["all"]["drift_rel"], 0.0)

                # W0 is the init: a resume must not overwrite it, and rebuilding it
                # from the seed (what the backfill does for old runs) must match.
                w0 = torch.load(os.path.join(out_dir, "w0.pt"))
                for k in w0:
                    torch.testing.assert_close(w0[k], w0_after_first[k], rtol=0, atol=0)
                rebuilt = rebuild_w0(final["args"])
                for k in w0:
                    torch.testing.assert_close(rebuilt[k], w0[k], rtol=0, atol=0)

                # Backfilling a run that logged drift live gives the same final numbers.
                res = backfill(out_dir)
                self.assertTrue(res["w0_matches_saved"])
                for layer in ("conv5", "fc"):
                    for key in ("drift_rel", "spec_norm"):
                        self.assertAlmostEqual(res["final"][layer][key],
                                               rows[-1]["rec_drift"][layer][key], places=6)

                # The gap sweep loads runs this way; channels=4 comes from final.json.
                # checkpoint.pt always exists (best.pt only once test acc > 0).
                model = load_run(out_dir, which="checkpoint.pt")
                self.assertIsInstance(model, LateReadout if readout == "late" else RecurrentCSNN)


def _opt_args(rec_lr=None, rec_wd=0.0, opt="adam", rec_anchor=False):
    return Namespace(opt=opt, lr=1e-3, rec_lr=rec_lr, rec_wd=rec_wd, rec_anchor=rec_anchor)


class OptimizerTests(unittest.TestCase):
    def test_default_is_one_group_without_decay(self):
        net = RecurrentCSNN(**SMALL)
        opt, trainable = train_recurrent.build_optimizer(net, _opt_args())
        self.assertEqual(len(opt.param_groups), 1)
        self.assertEqual(opt.param_groups[0]["weight_decay"], 0.0)
        self.assertEqual(len(trainable), len(list(net.parameters())))

    def test_rec_group_gets_own_lr_and_decay(self):
        net = RecurrentCSNN(**SMALL)
        opt, _ = train_recurrent.build_optimizer(net, _opt_args(rec_lr=1e-4, rec_wd=0.1))
        base, rec = opt.param_groups
        rec_weights = {id(m.rec.weight) for m in net.rec_modules().values()}
        self.assertEqual({id(p) for p in rec["params"]}, rec_weights)
        self.assertFalse(rec_weights & {id(p) for p in base["params"]})
        self.assertEqual((rec["lr"], rec["weight_decay"]), (1e-4, 0.1))
        self.assertEqual((base["lr"], base["weight_decay"]), (1e-3, 0.0))

    def test_decay_shrinks_only_w_rec(self):
        # With zero gradients Adam's update is zero, so only the decoupled decay
        # acts: W_rec scales by (1 - lr * wd) and every other weight stays put.
        net = RecurrentCSNN(**SMALL)
        opt, trainable = train_recurrent.build_optimizer(net, _opt_args(rec_wd=0.5))
        before = {id(p): p.detach().clone() for p in trainable}
        for p in trainable:
            p.grad = torch.zeros_like(p)
        opt.step()
        rec_weights = {id(m.rec.weight) for m in net.rec_modules().values()}
        for p in trainable:
            expected = before[id(p)] * (1 - 1e-3 * 0.5) if id(p) in rec_weights else before[id(p)]
            torch.testing.assert_close(p.detach(), expected)

    def test_frozen_ignores_rec_settings(self):
        net = RecurrentCSNN(freeze_rec=True, **SMALL)
        opt, _ = train_recurrent.build_optimizer(net, _opt_args(rec_lr=1e-4, rec_wd=0.1))
        self.assertEqual(len(opt.param_groups), 1)


class AnchoredDecayTests(unittest.TestCase):
    def _setup(self, wd=0.5):
        net = RecurrentCSNN(**SMALL)
        opt, trainable = train_recurrent.build_optimizer(net, _opt_args(rec_wd=wd, rec_anchor=True))
        w0 = rec_weights(net)
        anchors = {id(m.rec.weight): w0[n] for n, m in net.rec_modules().items()}
        return net, opt, trainable, w0, anchors

    def _step_with_zero_grads(self, opt, trainable, anchors):
        # Zero gradients: Adam's update is zero, so only the decay acts.
        for p in trainable:
            p.grad = torch.zeros_like(p)
        train_recurrent.apply_anchored_decay(opt, anchors)
        opt.step()

    def test_group_settings(self):
        _, opt, _, _, _ = self._setup()
        base, rec = opt.param_groups
        self.assertEqual((rec["weight_decay"], rec["anchor_wd"]), (0.0, 0.5))
        self.assertEqual(base["weight_decay"], 0.0)
        net = RecurrentCSNN(**SMALL)
        opt, _ = train_recurrent.build_optimizer(net, _opt_args(rec_wd=0.5))
        self.assertEqual((opt.param_groups[1]["weight_decay"], opt.param_groups[1]["anchor_wd"]),
                         (0.5, 0.0))

    def test_pulls_toward_w0_not_zero(self):
        # W = W0 + D  ->  W0 + D * (1 - lr * wd); every other weight stays put.
        net, opt, trainable, w0, anchors = self._setup()
        g = torch.Generator().manual_seed(3)
        deltas = {}
        with torch.no_grad():
            for n, m in net.rec_modules().items():
                deltas[n] = torch.randn(m.rec.weight.shape, generator=g)
                m.rec.weight.add_(deltas[n])
        rec_ids = set(anchors)
        before = {id(p): p.detach().clone() for p in trainable if id(p) not in rec_ids}
        self._step_with_zero_grads(opt, trainable, anchors)
        for n, m in net.rec_modules().items():
            torch.testing.assert_close(m.rec.weight.detach(), w0[n] + deltas[n] * (1 - 1e-3 * 0.5))
        for p in trainable:
            if id(p) not in rec_ids:
                torch.testing.assert_close(p.detach(), before[id(p)])

    def test_no_pull_at_w0(self):
        # At W = W0 anchored decay does nothing; plain decay would shrink W.
        net, opt, trainable, w0, anchors = self._setup()
        self._step_with_zero_grads(opt, trainable, anchors)
        for n, m in net.rec_modules().items():
            torch.testing.assert_close(m.rec.weight.detach(), w0[n])


class DriftTests(unittest.TestCase):
    def test_init_stats(self):
        net = RecurrentCSNN(rec_layers=("conv5", "fc"), **SMALL)
        stats = net_drift_stats(net, rec_weights(net))
        for layer in ("conv5", "fc"):
            self.assertAlmostEqual(stats[layer]["drift_rel"], 0.0)
            self.assertAlmostEqual(stats[layer]["cos_to_init"], 1.0)
        # Orthogonal x 0.5: every singular value and |eigenvalue| of fc is 0.5.
        self.assertAlmostEqual(stats["fc"]["spec_norm"], 0.5, places=5)
        self.assertAlmostEqual(stats["fc"]["radius"], 0.5, places=5)
        self.assertNotIn("radius", stats["conv5"])
        self.assertEqual(net.rec_map_sizes(), {"conv5": 8, "fc": None})

    def test_conv_spectral_norm_matches_dense_operator(self):
        torch.manual_seed(0)
        c, size = 4, 8
        w = torch.randn(c, c, 3, 3, dtype=torch.float64)
        basis = torch.eye(c * size * size, dtype=torch.float64).reshape(-1, c, size, size)
        dense = F.conv2d(basis, w, padding=1).reshape(c * size * size, -1).T
        exact = torch.linalg.matrix_norm(dense, ord=2).item()
        self.assertAlmostEqual(conv_spectral_norm(w, size) / exact, 1.0, places=4)

    def test_known_drift_values(self):
        a = torch.randn(6, 6)
        sizes = {"fc": None}
        doubled = drift_stats({"fc": 2 * a}, {"fc": a}, sizes)["fc"]
        self.assertAlmostEqual(doubled["drift_rel"], 1.0, places=6)
        self.assertAlmostEqual(doubled["cos_to_init"], 1.0, places=6)
        flipped = drift_stats({"fc": -a}, {"fc": a}, sizes)["fc"]
        self.assertAlmostEqual(flipped["drift_rel"], 2.0, places=6)
        self.assertAlmostEqual(flipped["cos_to_init"], -1.0, places=6)
        # Radius of a diagonal matrix is its largest |entry|; spec norm too.
        diag = drift_stats({"fc": torch.diag(torch.tensor([0.3, -0.9, 0.1]))},
                           {"fc": torch.eye(3)}, sizes)["fc"]
        self.assertAlmostEqual(diag["radius"], 0.9, places=6)
        self.assertAlmostEqual(diag["spec_norm"], 0.9, places=6)

    def test_zero_init_has_no_relative_drift(self):
        net = RecurrentCSNN(rec_init="zero", **SMALL)
        stats = net_drift_stats(net, rec_weights(net))
        self.assertIsNone(stats["fc"]["drift_rel"])
        self.assertIsNone(stats["fc"]["cos_to_init"])
        self.assertIsNone(stats["all"]["drift_rel"])
        self.assertEqual(stats["all"]["spec_norm_max"], 0.0)

    def test_does_not_touch_global_rng(self):
        # Logging every epoch must not change dropout masks or data order.
        net = RecurrentCSNN(**SMALL)
        w0 = rec_weights(net)
        state = torch.get_rng_state()
        net_drift_stats(net, w0)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))


class ProbeTests(unittest.TestCase):
    def test_probe_reports_every_layer_and_removes_hooks(self):
        net = RecurrentCSNN(**SMALL)
        loader = [(_frames(T=3, B=2, seed=s), torch.tensor([1, 4])) for s in range(2)]
        for model in (net, LateReadout(net, 2)):
            res = probe(model, loader, torch.device("cpu"))
            self.assertEqual(res["clips"], 4)
            self.assertTrue(0.0 <= res["accuracy"] <= 1.0)
            self.assertEqual(set(res["layers"]),
                             {"conv1", "conv2", "conv3", "conv4", "conv5", "fc", "out"})
            for label, row in res["layers"].items():
                self.assertTrue(0.0 <= row["firing_rate"] <= 1.0, label)
                self.assertEqual("rec_to_ff" in row, label in ("conv5", "fc"), label)
            self.assertAlmostEqual(res["spikes_per_clip"],
                                   sum(r["spikes_per_clip"] for r in res["layers"].values()))
            self.assertFalse(any(m._forward_hooks or m._forward_pre_hooks for m in net.modules()))


if __name__ == "__main__":
    unittest.main()

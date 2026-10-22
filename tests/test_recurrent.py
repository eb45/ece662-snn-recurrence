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
from spikingjelly.activation_based import functional
from torch.utils.data import DataLoader, TensorDataset

import train_recurrent
from models import CSNN
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
        torch.manual_seed(0)
        ctrl = CSNN(**SMALL).eval()
        rnet = RecurrentCSNN(rec_layers=("conv1", "conv5", "fc"), rec_init="zero", **SMALL).eval()
        missing, unexpected = rnet.load_state_dict(ctrl.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertTrue(missing and all(k.endswith("rec.weight") for k in missing))

        x = _frames(T=4)
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

        def make_args(output_dir, epochs, readout="mean"):
            return Namespace(
                root="unused", out=output_dir, epochs=epochs, batch_size=2, accum=1, T=2,
                channels=4, opt="adam", lr=1e-3, loss="mse", seed=17, num_workers=0,
                no_amp=True, no_resume=False, rec_layers=["conv5", "fc"],
                rec_init="orthogonal", rec_gain=0.5, freeze_rec=False,
                readout=readout, late_k=1, clip_grad=1.0,
            )

        for readout in ("mean", "late"):
            with self.subTest(readout=readout), tempfile.TemporaryDirectory() as out_dir:
                with patch.object(train_recurrent, "get_loaders", return_value=(loader, loader)):
                    with patch.object(train_recurrent, "parse_args",
                                      return_value=make_args(out_dir, 1, readout)):
                        train_recurrent.main()
                    with patch.object(train_recurrent, "parse_args",
                                      return_value=make_args(out_dir, 2, readout)):
                        train_recurrent.main()

                with open(os.path.join(out_dir, "metrics.jsonl")) as f:
                    rows = [json.loads(line) for line in f]
                self.assertEqual([r["epoch"] for r in rows], [0, 1])
                self.assertEqual(set(rows[0]["rec_weight_norm"]), {"conv5", "fc"})

                with open(os.path.join(out_dir, "final.json")) as f:
                    final = json.load(f)
                self.assertIsNotNone(final["final_test_acc"])

                # The gap sweep loads runs this way; channels=4 comes from final.json.
                # checkpoint.pt always exists (best.pt only once test acc > 0).
                model = load_run(out_dir, which="checkpoint.pt")
                self.assertIsInstance(model, LateReadout if readout == "late" else RecurrentCSNN)


if __name__ == "__main__":
    unittest.main()

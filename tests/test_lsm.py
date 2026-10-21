import json
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader, TensorDataset

import train_lsm
from lsm_model import BASELINE_LIF_NEURONS, LSM


def _time_first_collate(batch):
    frames = torch.stack([item[0] for item in batch], dim=0)
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return frames.transpose(0, 1).contiguous(), labels


class LSMModelTests(unittest.TestCase):
    def make_model(self):
        return LSM(
            num_classes=11,
            lif_neurons=64,
            readout_features=8,
            recurrent_degree=2,
            seed=17,
        )

    def test_uses_requested_lif_count_and_baseline_target(self):
        model = self.make_model()
        self.assertEqual(model.num_lif_neurons(), 64)
        self.assertEqual(BASELINE_LIF_NEURONS, 2_794_094)

    def test_output_shape_finite_loss_and_gradients(self):
        model = self.make_model()
        frames = torch.randint(0, 3, (3, 2, 2, 128, 128)).float()
        labels = torch.tensor([1, 4])

        output = model(frames)
        self.assertEqual(tuple(output.shape), (3, 2, 11))
        self.assertTrue(torch.isfinite(output).all())

        loss = torch.nn.functional.mse_loss(
            output.mean(0), torch.nn.functional.one_hot(labels, num_classes=11).float()
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_liquid_state_resets_at_each_forward(self):
        model = self.make_model().eval()
        frames = torch.randint(0, 3, (2, 1, 2, 128, 128)).float()

        with torch.no_grad():
            first_output = model(frames)
            second_output = model(frames)

        torch.testing.assert_close(first_output, second_output)

    def test_rejects_invalid_input_shape(self):
        model = self.make_model()
        with self.assertRaisesRegex(ValueError, r"\[T, B, 2, 128, 128\]"):
            model(torch.zeros(2, 1, 1, 128, 128))


class LSMTrainingTests(unittest.TestCase):
    def test_metrics_checkpoint_and_resume(self):
        frames = torch.randint(0, 3, (4, 2, 2, 128, 128)).float()
        labels = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        dataset = TensorDataset(frames, labels)
        loader = DataLoader(
            dataset, batch_size=2, shuffle=False, collate_fn=_time_first_collate
        )

        def make_args(output_dir, epochs, no_resume=False):
            return Namespace(
                root="unused",
                out=output_dir,
                epochs=epochs,
                batch_size=2,
                accum=1,
                T=2,
                lif_neurons=64,
                readout_features=8,
                recurrent_degree=2,
                tau=2.0,
                threshold=1.0,
                input_scale=0.1,
                recurrent_gain=0.25,
                opt="adam",
                lr=1e-3,
                loss="mse",
                seed=17,
                num_workers=0,
                no_amp=True,
                no_resume=no_resume,
            )

        with tempfile.TemporaryDirectory() as output_dir:
            with patch.object(train_lsm, "get_loaders", return_value=(loader, loader)):
                with patch.object(
                    train_lsm, "parse_args", return_value=make_args(output_dir, 1)
                ):
                    train_lsm.main()

                checkpoint_path = os.path.join(output_dir, "checkpoint.pt")
                metrics_path = os.path.join(output_dir, "metrics.jsonl")
                self.assertTrue(os.path.isfile(checkpoint_path))
                with open(metrics_path, encoding="utf-8") as metrics_file:
                    first_rows = [json.loads(line) for line in metrics_file]
                self.assertEqual(len(first_rows), 1)
                self.assertIn("test_acc", first_rows[0])

                with patch.object(
                    train_lsm, "parse_args", return_value=make_args(output_dir, 2)
                ):
                    train_lsm.main()

            with open(metrics_path, encoding="utf-8") as metrics_file:
                rows = [json.loads(line) for line in metrics_file]
            self.assertEqual([row["epoch"] for row in rows], [0, 1])


if __name__ == "__main__":
    unittest.main()

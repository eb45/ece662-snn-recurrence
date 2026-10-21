"""Fixed sparse liquid-state machine for DVS128 Gesture.

The liquid is a recurrent population of heterogeneous LIF neurons with fixed
random input and recurrent connections. A trainable linear readout operates on
pooled liquid activity at each time step.

Input:  [T, B, 2, 128, 128]
Output: [T, B, num_classes]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


BASELINE_LIF_NEURONS = 2_794_094


class LSM(nn.Module):
    """Sparse fixed reservoir with a trainable readout."""

    def __init__(
        self,
        num_classes: int = 11,
        lif_neurons: int = BASELINE_LIF_NEURONS,
        readout_features: int = 512,
        tau: float = 2.0,
        threshold: float = 1.0,
        input_scale: float = 0.1,
        recurrent_gain: float = 0.25,
        recurrent_degree: int = 4,
        seed: int = 2026,
    ):
        super().__init__()
        if num_classes <= 0 or lif_neurons <= 0 or readout_features <= 0:
            raise ValueError("num_classes, lif_neurons, and readout_features must be positive")
        if recurrent_degree <= 0:
            raise ValueError("recurrent_degree must be positive")
        if tau <= 1.0:
            raise ValueError("tau must be greater than 1")
        if threshold <= 0.0 or input_scale < 0.0 or recurrent_gain < 0.0:
            raise ValueError("threshold must be positive; input_scale and recurrent_gain cannot be negative")

        self.num_classes = num_classes
        self.lif_neurons = lif_neurons
        self.readout_features = readout_features
        self.tau = tau
        self.threshold = threshold
        self.input_scale = input_scale
        self.recurrent_gain = recurrent_gain
        self.recurrent_degree = recurrent_degree
        self.seed = seed

        # Generate fixed wiring from a local RNG so initialization does not
        # disturb the global RNG used by the trainable readout.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        input_features = 2 * 128 * 128
        input_indices = torch.randint(
            input_features, (lif_neurons,), generator=generator, dtype=torch.long
        )
        recurrent_indices = torch.randint(
            lif_neurons,
            (recurrent_degree, lif_neurons),
            generator=generator,
            dtype=torch.long,
        )
        tau_values = tau * (
            0.8 + 0.4 * torch.rand(lif_neurons, generator=generator)
        )
        threshold_values = threshold * (
            0.9 + 0.2 * torch.rand(lif_neurons, generator=generator)
        )

        # Wiring is reproducibly regenerated from the saved model arguments,
        # avoiding large fixed buffers in every checkpoint.
        self.register_buffer("input_indices", input_indices, persistent=False)
        self.register_buffer("recurrent_indices", recurrent_indices, persistent=False)
        self.register_buffer("tau_values", tau_values, persistent=False)
        self.register_buffer("threshold_values", threshold_values, persistent=False)

        self.readout = nn.Linear(readout_features, num_classes)

    def _liquid_features(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.ndim != 5 or tuple(frames.shape[2:]) != (2, 128, 128):
            raise ValueError(
                "frames must have shape [T, B, 2, 128, 128], "
                f"got {tuple(frames.shape)}"
            )

        time_steps, batch_size = frames.shape[:2]
        with torch.no_grad():
            membrane = frames.new_zeros((batch_size, self.lif_neurons), dtype=torch.float32)
            spikes = torch.zeros_like(membrane)
            features = []
            recurrent_weight = self.recurrent_gain / self.recurrent_degree

            for time_index in range(time_steps):
                inputs = frames[time_index].reshape(batch_size, -1).float()
                current = inputs.index_select(1, self.input_indices) * self.input_scale
                for source_indices in self.recurrent_indices:
                    current.add_(
                        spikes.index_select(1, source_indices),
                        alpha=recurrent_weight,
                    )

                tau = self.tau_values.unsqueeze(0)
                threshold = self.threshold_values.unsqueeze(0)
                membrane.mul_(1.0 - 1.0 / tau).add_(current)
                spikes = (membrane >= threshold).to(dtype=membrane.dtype)
                membrane.sub_(spikes * threshold)

                pooled = F.adaptive_avg_pool1d(
                    spikes.unsqueeze(1), self.readout_features
                ).squeeze(1)
                features.append(pooled)

        return torch.stack(features, dim=0)

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        features = self._liquid_features(frames)
        time_steps, batch_size, _ = features.shape
        logits = self.readout(features.reshape(time_steps * batch_size, -1))
        return logits.reshape(time_steps, batch_size, self.num_classes)

    def num_params(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def num_lif_neurons(self) -> int:
        """Return liquid LIF neurons per time step (exactly the configured count)."""
        return self.lif_neurons

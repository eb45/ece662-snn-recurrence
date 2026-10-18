"""
Feed-forward spiking CNN control for DVS128 Gesture.

Follows the SpikingJelly "Classify DVS Gesture" tutorial: 5 blocks of
Conv-BN-LIF-MaxPool, then a spiking MLP head with a voting layer.

The only memory in this network is the LIF membrane leak. There are no
recurrent connections, which is what makes it the control arm.

Input:  [T, B, 2, 128, 128]  (time first, multi-step mode)
Output: [T, B, num_classes]  (spikes per time step; average over T for the
        firing rate used by the loss and for the prediction)
"""

import torch.nn as nn
from spikingjelly.activation_based import functional, layer, neuron, surrogate


def make_lif(tau: float = 2.0):
    # Same neuron settings as the tutorial. Keep this in one place so the
    # recurrent arm can reuse it and the "same neuron model" claim holds.
    return neuron.LIFNode(tau=tau, surrogate_function=surrogate.ATan(), detach_reset=True)


class CSNN(nn.Module):
    def __init__(self, channels: int = 128, num_blocks: int = 5, num_classes: int = 11,
                 voting: int = 10, in_channels: int = 2, input_size: int = 128,
                 hidden: int = 512, dropout: float = 0.5, tau: float = 2.0):
        super().__init__()
        self.channels = channels
        self.num_blocks = num_blocks
        self.num_classes = num_classes
        self.voting = voting
        self.input_size = input_size
        self.hidden = hidden

        conv = []
        c_in = in_channels
        for _ in range(num_blocks):
            conv += [
                layer.Conv2d(c_in, channels, kernel_size=3, padding=1, bias=False),
                layer.BatchNorm2d(channels),
                make_lif(tau),
                layer.MaxPool2d(2, 2),
            ]
            c_in = channels
        self.conv_stack = nn.Sequential(*conv)

        final_size = input_size // (2 ** num_blocks)  # 128 / 32 = 4
        self.head = nn.Sequential(
            layer.Flatten(),
            layer.Dropout(dropout),
            layer.Linear(channels * final_size * final_size, hidden),
            make_lif(tau),
            layer.Dropout(dropout),
            layer.Linear(hidden, num_classes * voting),
            make_lif(tau),
            layer.VotingLayer(voting),
        )

        functional.set_step_mode(self, "m")

    def forward(self, x):
        # x: [T, B, C, H, W]
        return self.head(self.conv_stack(x))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_lif_neurons(self) -> int:
        """LIF neurons per time step. Use this to match neuron counts across arms."""
        total, size = 0, self.input_size
        for _ in range(self.num_blocks):
            total += self.channels * size * size  # LIF sits before the pool
            size //= 2
        total += self.hidden                      # hidden spiking layer
        total += self.num_classes * self.voting   # output spiking layer
        return total

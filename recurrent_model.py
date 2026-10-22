"""
Deep recurrent spiking CNN for DVS128 Gesture (the "ours" arm).

This is the feed-forward control (models.CSNN) plus trainable recurrent
weights at one or more LIF layers. At a recurrent layer the neuron input is

    I_t = BN(Conv(x_t)) + W_rec * s_{t-1}

so spikes at t feed back at t+1. Everything else (conv stack, BatchNorm, LIF
settings, head, voting layer, neuron count) is the control, unchanged.
Recurrence adds synapses, not neurons. With W_rec = 0 the network computes
exactly what CSNN computes, which tests/test_recurrent.py checks.

Recurrent layers are named by where the LIF sits:
    conv1..conv5  LIF in conv block b (before its MaxPool); W_rec is a 3x3 conv, C -> C
    fc            the 512-unit hidden LIF in the head; W_rec is a 512 x 512 linear map

Input:  [T, B, 2, 128, 128]  (any T, so gapped clips work)
Output: [T, B, num_classes]
"""

import json
import os

import torch
import torch.nn as nn

from models import CSNN, make_lif

REC_LAYER_NAMES = ("conv1", "conv2", "conv3", "conv4", "conv5", "fc")
REC_INITS = ("orthogonal", "zero")


class RecurrentLIF(nn.Module):
    """LIF layer whose input at step t also gets rec(spikes at t-1).

    The feed-forward current for all T steps arrives precomputed, so the conv
    and BatchNorm before this layer run in multi-step mode exactly as in the
    control (BN sees the same [T*B] statistics). Only the recurrent term needs
    the per-step loop.
    """

    def __init__(self, rec: nn.Module, tau: float = 2.0):
        super().__init__()
        self.rec = rec          # nn.Linear or nn.Conv2d, bias=False so rec(0) == 0
        self.lif = make_lif(tau)

    def forward(self, x_seq):
        # functional.set_step_mode(model, "m") flips every neuron to multi-step,
        # including this one. The loop below feeds one step at a time, so force
        # single-step here instead of trusting whoever called set_step_mode last.
        self.lif.step_mode = "s"
        s = torch.zeros_like(x_seq[0])
        spikes = []
        for x_t in x_seq:
            s = self.lif(x_t + self.rec(s))
            spikes.append(s)
        return torch.stack(spikes)


def _init_rec(weight: torch.Tensor, init: str, gain: float) -> None:
    with torch.no_grad():
        if init == "zero":
            weight.zero_()
        else:
            # Orthogonal rows scaled by gain (Yin et al. 2021 use orthogonal
            # recurrent init). For fc this makes the spectral radius = gain; for
            # conv the weight is flattened to [C, C*3*3] and each output channel's
            # kernel gets norm = gain.
            nn.init.orthogonal_(weight, gain=gain)


class RecurrentCSNN(CSNN):
    def __init__(self, rec_layers=("conv5", "fc"), rec_init: str = "orthogonal",
                 rec_gain: float = 0.5, freeze_rec: bool = False, tau: float = 2.0,
                 **csnn_kwargs):
        # Build the control first so its weights are drawn from the RNG in the
        # same order as CSNN. With the same seed the shared weights start equal.
        super().__init__(tau=tau, **csnn_kwargs)

        rec_layers = tuple(dict.fromkeys(rec_layers))  # drop duplicates, keep order
        if not rec_layers:
            raise ValueError("rec_layers is empty; use models.CSNN for no recurrence")
        valid = [f"conv{b}" for b in range(1, self.num_blocks + 1)] + ["fc"]
        for name in rec_layers:
            if name not in valid:
                raise ValueError(f"unknown recurrent layer {name!r}; choose from {valid}")
        if rec_init not in REC_INITS:
            raise ValueError(f"rec_init must be one of {REC_INITS}")
        if freeze_rec and rec_init == "zero":
            raise ValueError("frozen zero recurrence is just the control; use a random init")

        self.rec_layers = rec_layers
        self.rec_init = rec_init
        self.rec_gain = rec_gain
        self.freeze_rec = freeze_rec

        for name in rec_layers:
            if name == "fc":
                rec = nn.Linear(self.hidden, self.hidden, bias=False)
            else:
                rec = nn.Conv2d(self.channels, self.channels, kernel_size=3, padding=1, bias=False)
            _init_rec(rec.weight, rec_init, rec_gain)
            rec.weight.requires_grad_(not freeze_rec)
            # Swap the control's LIF in place so module indices (and therefore
            # state_dict keys) stay the same as CSNN's.
            container, idx = self._lif_slot(name)
            container[idx] = RecurrentLIF(rec, tau)

    def _lif_slot(self, name: str):
        # Conv block b is [Conv, BN, LIF, Pool] at indices 4(b-1) .. 4(b-1)+3.
        # Head is [Flatten, Dropout, Linear, LIF, Dropout, Linear, LIF, Voting].
        if name == "fc":
            return self.head, 3
        b = int(name[len("conv"):])
        return self.conv_stack, 4 * (b - 1) + 2

    def rec_modules(self) -> dict:
        """{layer name: RecurrentLIF}, in the order given at construction."""
        out = {}
        for name in self.rec_layers:
            container, idx = self._lif_slot(name)
            out[name] = container[idx]
        return out

    def num_rec_params(self) -> int:
        return sum(m.rec.weight.numel() for m in self.rec_modules().values())

    def num_total_params(self) -> int:
        """All parameters, including frozen recurrent weights."""
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def rec_weight_norms(self) -> dict:
        """Frobenius norm of each W_rec, logged per epoch to see whether it is used."""
        return {name: round(m.rec.weight.float().norm().item(), 4)
                for name, m in self.rec_modules().items()}


class LateReadout(nn.Module):
    """Keep only the last k output steps.

    evaluate() and the training loop average model output over time. Wrapping a
    model in this makes that average cover only the end of the clip, so the
    prediction has to come from what the network still holds at the end rather
    than from spikes summed over the whole clip (see Cramer et al. 2022 on
    last-step losses). Works for any arm that returns [T, B, num_classes]. It
    has no parameters, so save and load the inner model's state_dict.
    """

    def __init__(self, model: nn.Module, k: int):
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        self.model = model
        self.k = k

    def forward(self, x):
        return self.model(x)[-self.k:]


def build_from_args(args: dict, num_classes: int = 11) -> RecurrentCSNN:
    """Rebuild the network from the args saved in a run's final.json."""
    return RecurrentCSNN(
        rec_layers=args["rec_layers"], rec_init=args["rec_init"], rec_gain=args["rec_gain"],
        freeze_rec=args["freeze_rec"], channels=args["channels"], num_classes=num_classes,
    )


def load_run(run_dir: str, device="cpu", which: str = "best.pt"):
    """Load a finished run for evaluation (e.g. the gap sweep).

    Returns the model wrapped in LateReadout if the run was trained that way,
    so evaluate(model, loader, device, transform=...) scores it the same way
    it was trained.
    """
    with open(os.path.join(run_dir, "final.json")) as f:
        args = json.load(f)["args"]
    net = build_from_args(args)
    state = torch.load(os.path.join(run_dir, which), map_location=device)
    if "model" in state and "opt" in state:  # checkpoint.pt rather than best.pt
        state = state["model"]
    net.load_state_dict(state)
    net.to(device).eval()
    if args["readout"] == "late":
        return LateReadout(net, args["late_k"])
    return net

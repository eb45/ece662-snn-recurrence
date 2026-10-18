"""
Shared evaluation for all three arms (control, LSM, recurrent SNN).

Every arm is scored by this one function so the numbers are comparable.

    from evaluate import evaluate
    acc = evaluate(model, test_loader, device)

Any model works as long as model(frames) takes [T, B, 2, 128, 128] and
returns [T, B, num_classes].

`transform` is an optional function applied to each batch of frames before
the model sees it. It is there for the gap-insertion test later, e.g.
transform=lambda x: insert_gap(x, start=8, length=16).
"""

import torch
from spikingjelly.activation_based import functional


@torch.no_grad()
def evaluate(model, loader, device, transform=None, amp: bool = False) -> float:
    """Top-1 accuracy over a loader. Resets neuron state after every batch."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    correct, total = 0, 0

    for frames, labels in loader:
        frames = frames.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if transform is not None:
            frames = transform(frames)

        with torch.autocast(device_type=device.type, enabled=use_amp):
            out = model(frames).mean(0)  # average spikes over time -> [B, num_classes]

        correct += (out.argmax(dim=1) == labels).sum().item()
        total += labels.numel()

        # Stale membrane potentials must not leak into the next batch.
        functional.reset_net(model)

    return correct / total

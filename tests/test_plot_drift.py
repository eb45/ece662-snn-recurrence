"""lr_curves groups runs into one accuracy-vs-W_rec-lr curve per placement."""

import os
import sys

import matplotlib
import pytest

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import plot_drift  # noqa: E402


def row(place, cond, lr, acc, seed=2026, main=True, dataset="dvs128", epochs=64):
    return {"run": f"{place}_{cond}_{lr}_{seed}", "placement": place, "condition": cond,
            "rec_lr": lr, "rec_wd": 0.0, "seed": seed, "main_setting": main,
            "dataset": dataset, "epochs_done": epochs, "final_test_acc": acc}


def test_lr_curves_groups_and_averages_seeds():
    rows = [
        row("fc", "lr grid", 1e-3, 0.80, seed=2026),
        row("fc", "lr grid", 1e-3, 0.90, seed=1),
        row("fc", "lr grid", 1e-4, 0.92),
        row("fc", "frozen", 1e-3, 0.85),
        row("conv5", "lr grid", 3e-5, 0.70),
        # Each of these is left out.
        row("fc", "lr grid", 3e-4, 0.10, main=False),          # off the main setting
        row("fc", "lr grid", 3e-4, 0.10, dataset="dvsgc"),     # other dataset
        row("fc", "lr grid", 3e-4, 0.10, epochs=2),            # debug run
        row("fc", "anchored", 1e-3, 0.10),                     # weight-decay runs
    ]
    c = plot_drift.lr_curves(rows)

    assert set(c) == {"fc", "conv5"}
    fc = c["fc"]["lr"]
    assert [p[0] for p in fc] == [1e-4, 1e-3]  # sorted by lr
    lr_1e3 = fc[1]
    assert lr_1e3[1] == pytest.approx(85.0)
    assert lr_1e3[2] == pytest.approx(7.0711, abs=1e-3)
    assert lr_1e3[3] == 2
    assert fc[0][2] == 0.0  # single seed: no spread
    assert c["fc"]["frozen"] == (pytest.approx(85.0), 0.0, 1)
    assert c["conv5"]["frozen"] is None


def test_plot_lr_writes_file(tmp_path):
    rows = [row("fc", "lr grid", lr, 0.9) for lr in (3e-5, 1e-4, 3e-4, 1e-3)]
    rows += [row("conv5+fc", "frozen", 1e-3, 0.92)]
    out = tmp_path / "lr.png"
    plot_drift.plot_lr(rows, 88.0, str(out))
    assert out.stat().st_size > 0

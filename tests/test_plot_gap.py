import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from plot_gap import load_gap_scores, mean_std


def _write_run(root, name, ds, arm, gaps, epochs=64):
    os.makedirs(os.path.join(root, "runs", name), exist_ok=True)
    with open(os.path.join(root, "runs", name, "gap.json"), "w") as f:
        json.dump({"gap": {str(k): v for k, v in gaps.items()}, "chance": 0.1}, f)
    return {"run": name, "dataset": ds, "arm": arm, "epochs_done": epochs, "epochs_planned": 64}


def test_load_gap_scores_groups_seeds_and_skips_short_runs(tmp_path):
    root = str(tmp_path)
    rows = [
        _write_run(root, "a_s1", "dvs128", "A", {0: 0.9, 8: 0.8}),
        _write_run(root, "a_s2", "dvs128", "A", {0: 0.8, 8: 0.6}),
        _write_run(root, "a_debug", "dvs128", "A", {0: 0.1, 8: 0.1}, epochs=2),
        _write_run(root, "grid", "dvs128", "", {0: 0.5, 8: 0.5}),
        _write_run(root, "other", "dvsgc-81p", "A", {0: 0.5, 8: 0.5}),
    ]
    with open(os.path.join(root, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    scores, chance = load_gap_scores(root, ["dvs128"])
    assert set(scores) == {("dvs128", "A")}
    assert scores[("dvs128", "A")][8] == [0.8, 0.6]
    assert chance["dvs128"] == 0.1
    m, s = mean_std(scores[("dvs128", "A")][8])
    assert abs(m - 0.7) < 1e-9 and s > 0


def test_relative_divides_each_seed_by_its_own_gap0():
    from plot_gap import relative
    rel = relative({0: [0.8, 0.5], 8: [0.4, 0.5]})
    assert rel[0] == [1.0, 1.0]
    assert rel[8] == [0.5, 1.0]

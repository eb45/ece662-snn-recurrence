# Results

Every number in the final report and poster comes from this folder.

## Layout

| Path | What it is |
| --- | --- |
| `runs/<run name>/` | Copies of each run's `final.json`, `metrics.jsonl` (one line per epoch), `gap.json`, and `spikes.json`. No checkpoints. |
| `summary.csv` | One row per run: dataset, arm, W_rec placement and learning setting, seed, final and best test accuracy, W_rec drift, gap / reverse / shuffle accuracy, spike counts. |
| `main_table.csv` | Every planned (dataset, arm), A to E plus LTC: final-epoch accuracy per seed, mean and std, gap-8 accuracy, W_rec drift. Seeds not run yet say `to run (<owner>)`. |
| `grid_table.csv` | DVS128 seed-2026 grid: W_rec placement (fc, conv5, conv5+fc) x learning setting, with final accuracy and drift. Cells not run yet say `to run (August)`. |

Both CSVs are rebuilt from `runs/` by `collect_results.py`, so never edit them by hand.

## Adding results

On DCC, from the repo root:

```
python collect_results.py /work/eb408/ece662_runs/*/
git add results && git commit -m "Add results" && git push
```

Run folders without `final.json` (still training or timed out) are skipped. To add someone else's runs, point the script at their run folders (needs read access), or have them run it and push.

## Arms

| Arm | Model |
| --- | --- |
| A | Feed-forward control, no W_rec |
| B | W_rec at fc, frozen at its random init |
| C | W_rec at fc, learned at the normal lr (1e-3) |
| D | W_rec at fc, learned at lr 1e-4 |
| E | W_rec at fc, anchored decay toward its init, wd 1.0 |

Other recurrent runs (the placement x learning-rate grid, conv5+fc ablations) have a blank arm and are labeled by placement and learning setting.

## Conventions

- **Final-epoch accuracy is the main number.** The best epoch is picked on the test set, so it is optimistic. `best_test_acc` is kept for reference.
- **Seeds:** 2026, 1, and 2.
- **Datasets:** `dvs128` (DVS128 Gesture, 11 classes), `dvsgc-81p` (DVS-GC, 4 gestures, predictable timing, 81 classes), `dvsgc-96u` (DVS-GC, 6 gestures, unpredictable timing, 96 classes).

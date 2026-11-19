# ECE662 SNN recurrence project

Team project (Emma Bennett, Anna Carpenter, August Hao) comparing a feed-forward SNN control, a fixed-reservoir LSM, and a learned-recurrent SNN on event-based data. Paper scope is DVS128 Gesture only; I am scouting extra datasets to decide whether to add them.

## My task (branch: datasets)
Prep three extra datasets and test them with the feed-forward control only:
1. DVS-Gesture-Chain (pip install dvsgc, built from the DVS128 Gesture data we already have)
2. DVS-Lip (SpikingJelly 2.0.0 DVSLip, 128x128, 100 classes)
3. SHD (SpikingJelly 2.0.0 or Tonic, 700 channels, 20 classes, no image structure)

For each: loader returning time-first batches, cache + sanity check, 2-epoch debug run, short real run, time-shuffle test, and a row in a results table (dataset, T, FF accuracy, chance, shuffle drop, time/epoch, peak GPU memory, cache size).

## Where things run
- Laptop: writing code and CPU unit tests only. No GPU, no real datasets.
- Duke Compute Cluster (DCC): all real data and training. I pull this branch there and run jobs.
  - DVS128 Gesture root: /work/eb408/dvs128_gesture
  - Run outputs: /work/eb408/ece662_runs/<run_name>
  - Conda env: rapids
  - GPU: RTX 2080 Ti, 11 GB. Control at T=16 uses ~3.9 GB, ~23 s/epoch.
- Anything longer than a few minutes goes through an sbatch script (copy train_control.sbatch). Runs checkpoint every epoch and resume after preemption.

## Rules for shared code
- dvs_data.py, models.py, evaluate.py are used by all three arms. Keep DVS128 Gesture behavior identical by default (T=16, split_by="number", 11 classes, [T, B, 2, 128, 128]). Add options; don't change defaults.
- Put new dataset code in new files (e.g. datasets_extra.py) where possible.
- Batches are time-first: frames [T, B, ...], labels [B]. Models return [T, B, num_classes]; evaluate averages spikes over time.
- Use the shared make_lif (tau=2) for any new model. SHD needs a separate dense feed-forward control built from make_lif.
- Don't touch recurrent_model.py, LSM code, or their tests unless asked.

## Testing
- CPU unit tests under tests/ with small synthetic tensors, no downloads.
- Run python -m pytest tests -q before committing.
- For anything only checkable on DCC, give me the exact command to run there.

## Style
- Short, clear commits. No em dashes in docs or comments.

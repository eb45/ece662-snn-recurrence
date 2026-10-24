# Recurrent SNN arm

The "ours" arm of the project: the feed-forward control with **trainable
recurrent connections** at one or two LIF layers, so spikes at step *t* feed
back at step *t + 1*. Everything else is the control.

Status: written and reviewed by reading, **not yet run**. Run the tests and a
2-epoch debug job before the first full run (see [Checklist](#checklist-before-the-first-full-run)).

## Files

| File | What it is |
|---|---|
| `recurrent_model.py` | `RecurrentLIF`, `RecurrentCSNN`, `LateReadout`, and `load_run()` for evaluation scripts |
| `train_recurrent.py` | Training script. Same loop as `train_control.py` plus recurrence/readout flags |
| `train_recurrent.sbatch` | DCC launcher. Variants are picked with environment variables |
| `tests/test_recurrent.py` | CPU unit tests, including "zero recurrence == control" |
| `probe_recurrent.py` | Firing rates, spikes per clip, and recurrent-vs-feed-forward drive for trained runs |

No shared file was changed. `dvs_data.py`, `models.py`, and `evaluate.py` are
imported as-is.

```
dvs_data.get_loaders ──► [T, B, 2, 128, 128] ──► RecurrentCSNN ──► [T, B, 11]
                                                   │  (subclass of models.CSNN,
                                                   │   reuses models.make_lif)
                                                   ▼
               optional LateReadout(k) ──► keeps the last k steps
                                                   ▼
          train_recurrent.py loss / evaluate.evaluate()  (mean over remaining steps)
                                                   ▼
          <out>/metrics.jsonl, best.pt, checkpoint.pt, final.json
                                                   ▼
   plot_metrics.py (curves)   recurrent_model.load_run() ──► gap sweep / spike counts
```

## Architecture

At a recurrent layer the LIF input is

```
I_t = BN(Conv(x_t)) + W_rec · s_{t-1}          (s_{-1} = 0)
```

- The conv and BatchNorm run on all T steps at once, exactly as in the
  control, so BatchNorm statistics are unchanged. Only the recurrent term is
  computed in a per-step loop inside `RecurrentLIF`.
- `W_rec` has no bias. With `W_rec = 0` the network is the control, which the
  tests check numerically.
- `RecurrentCSNN` builds a `CSNN` and swaps the chosen LIF modules for
  `RecurrentLIF` in place. Module indices, and therefore state_dict keys, match
  the control's, so a control checkpoint loads with `strict=False` (only the
  `rec.weight` keys are missing).
- The CSNN weights are drawn first, so with the same seed the shared layers
  start from the same values as the control's.

| Layer name | Where | `W_rec` | Recurrent params |
|---|---|---|---|
| `conv1` … `conv5` | LIF of conv block b, before its MaxPool | 3×3 conv, 128 → 128 | 147,456 each |
| `fc` | 512-unit hidden LIF in the head | 512 × 512 linear | 262,144 |

Map sizes at the LIF are 128, 64, 32, 16, 8 for blocks 1–5. Early blocks are
the most expensive to make recurrent.

### Counts for the report

Neuron count is identical to the control in every variant, because recurrence
adds synapses, not neurons.

| Variant | Trainable params | LIF neurons / step |
|---|---|---|
| Control (`models.CSNN`) | 1,698,926 | 2,794,094 |
| `fc` | 1,961,070 | 2,794,094 |
| `conv5` | 1,846,382 | 2,794,094 |
| `conv5 fc` (default) | 2,108,526 | 2,794,094 |
| `conv5 fc --freeze-rec` | 1,698,926 trainable + 409,600 frozen | 2,794,094 |

### Initialization

`--rec-init orthogonal --rec-gain 0.5` (default) uses an orthogonal matrix
scaled by the gain, following Yin et al. (2021). For `fc` the spectral radius
is 0.5, so recurrent activity decays rather than explodes at the start. For
conv each output channel's 3×3×128 kernel has norm 0.5. The recurrent current
bypasses BatchNorm, which is why the scale is kept below 1.

`--rec-init zero` starts exactly at the control. The gradient of `W_rec` is
still non-zero, since it depends on the spikes, not on `W_rec`, so it can learn
from there. It is not allowed together with `--freeze-rec`, which would just
be the control.

## Running

Use the same environment as the control (`requirements.txt`: torch 2.6,
spikingjelly 2.0.0).

```bash
# Tests (CPU, no dataset needed), from the repo root
python -m unittest tests.test_recurrent -v

# Debug: 2 epochs, check time/epoch and peak memory against the control (~23 s, 3.9 GB)
EPOCHS=2 RUN_NAME=rec_debug sbatch --time=00:45:00 train_recurrent.sbatch

# Main arm: learned recurrence at conv5 + fc
sbatch train_recurrent.sbatch                       # -> rec_conv5_fc_s2026

# Variants
REC_LAYERS="fc" sbatch train_recurrent.sbatch       # -> rec_fc_s2026
FREEZE_REC=1 sbatch train_recurrent.sbatch          # -> rec_conv5_fc_frozen_s2026
READOUT=late sbatch train_recurrent.sbatch          # -> rec_conv5_fc_late_s2026
SEED=7 sbatch train_recurrent.sbatch                # extra seeds
```

Direct Python call (Colab or an interactive node):

```bash
python train_recurrent.py --root /path/to/dvs128_gesture --out runs/rec_conv5_fc \
    --rec-layers conv5 fc --epochs 64
```

### Flags added on top of `train_control.py`

| Flag | Default | Meaning |
|---|---|---|
| `--rec-layers` | `conv5 fc` | Which LIF layers get `W_rec` |
| `--rec-init` | `orthogonal` | `orthogonal` or `zero` |
| `--rec-gain` | `0.5` | Scale of the orthogonal init |
| `--freeze-rec` | off | Keep `W_rec` at its random init (random-recurrence arm) |
| `--readout` | `mean` | `mean` = average over all steps (control). `late` = last `--late-k` steps only |
| `--late-k` | `4` | Steps used by the late readout |
| `--clip-grad` | `0` (off) | Gradient-norm clipping. If used, use it for every arm |
| `--rec-lr` | same as `--lr` | Learning rate for `W_rec` only |
| `--rec-wd` | `0` (off) | Decoupled (AdamW-style) weight decay on `W_rec` only |

sbatch equivalents: `REC_LAYERS`, `REC_INIT`, `REC_GAIN`, `FREEZE_REC=1`,
`READOUT`, `LATE_K`, `CLIP_GRAD`, `REC_LR`, `REC_WD`. The default run name
encodes every non-default setting (e.g. `rec_fc_zero_wd1.0_s2026`), so variants
never share an output folder or resume each other's checkpoint.

The optimizer is `AdamW`. With `weight_decay=0`, which applies to every
non-recurrent weight, it is exactly Adam, as in the control. Only `W_rec`
can get decay. AdamW's decay shrinks weights by `lr × wd` per step,
independent of Adam's gradient scaling. With about 4,700 steps and a cosine
schedule, `wd = 0.1` shrinks an unused weight to about 80%, and `wd = 1.0` to
about 10%. So useful values are larger than the 1e-4-style L2 values used with
plain Adam.

### Outputs

Same files as the control, with a few extra fields:

- `metrics.jsonl` rows also have `rec_weight_norm`, the norm of each `W_rec`
  per epoch. If it stays near its starting value, the network is not using
  recurrence. `plot_metrics.py` ignores the extra key.
- `final.json` also has `final_test_acc` (last epoch, which the report says we
  will use), `total_params`, and `rec_params`.
- `best.pt` / `checkpoint.pt` hold the `RecurrentCSNN` weights (never the
  `LateReadout` wrapper), so keys match the control's.

### Loading a run for the gap sweep or spike counts

```python
from recurrent_model import load_run
from evaluate import evaluate

model = load_run("/work/eb408/ece662_runs/rec_conv5_fc_s2026", device)   # rebuilt from final.json
acc = evaluate(model, test_loader, device, transform=lambda x: insert_gap(x, start=8, length=16))
```

`load_run` returns the model wrapped in `LateReadout` if it was trained with
`--readout late`, so it is scored the way it was trained. It needs
`final.json`, which is written when training finishes.

### Diagnostics: `probe_recurrent.py`

```bash
python probe_recurrent.py --root /hpc/home/ah670/dvs128_gesture \
    --runs /work/ah670/ece662_runs/rec_fc_s2026 /work/ah670/ece662_runs/rec_conv5_fc_frozen_s2026
```

For each run, on the test split, it reports:

- **Accuracy** of `best.pt`. This should match `best_test_acc`, give or take a
  clip because of mixed precision, which also confirms loading works.
- **Firing rate per spiking layer** (`conv1`–`conv5`, `fc`, `out`).
- **Spikes per clip**, per layer and in total.
- **`rec/ff`** for each recurrent layer: the RMS of `W_rec·s_{t-1}` divided by
  the RMS of the feed-forward input. Near 0 means recurrence is barely used;
  around 1 or more means the layer is driven as much by its own past spikes as
  by the input.

Results are also saved to `<run>/probe.json`. Runs without `rec_layers` in
`final.json` are loaded as the control, so the control can be probed for
comparison.

## Results so far (seed 2026, mean readout, 288 test clips)

| Run | Train acc | Test best / final | `|W_rec|` start → end |
|---|---|---|---|
| Control (`models.CSNN`) | ~100% | 94.1 / 92.4 | – |
| conv5+fc, frozen random | 99.9% | 93.8 / 92.4 | fixed (5.7, 11.3) |
| fc, learned | 99.6% | 89.2 / 88.2 | 11.3 → 31.7 |
| conv5+fc, learned, zero init | 99.7% | 88.2 / 87.2 | 0 → (19.2, 27.8) |
| conv5+fc, learned | 99.7% | 86.5 / 85.1 | (5.7, 11.3) → (19.1, 29.4) |
| conv5+fc, learned, clip 1.0 | – | 86.5 / 85.1 (identical: clipping never triggered) | same |

**Reading.** Learned recurrence fits the training people as well as the control,
but generalizes 5–8 points worse to unseen people, and more of it overfits more.
`W_rec` grows 2.5–3.5× and nothing counteracts that growth. Random fixed recurrence
costs nothing. Next: `--rec-wd` / `--rec-lr` to regularize `W_rec`, and
`probe_recurrent.py` to measure how much each layer relies on recurrence.

## Differences

### From the feed-forward control (`models.py`, `train_control.py`)

What matches: data, split, T = 16, conv stack, BatchNorm, LIF (`make_lif`,
τ = 2, ATan, detached reset), dropout, voting head, neuron count, Adam 1e-3,
cosine schedule, MSE on firing rate, 64 epochs, batch 8 × accum 2, AMP,
seed, checkpoint/resume logic, `evaluate()`.

What differs:

1. **Model.** `W_rec` at the chosen layers, which adds 409,600 params by default.
2. **Speed.** Recurrent LIF layers run a Python loop over T in single-step
   mode. Expect it to be somewhat slower than the control, and measure it in
   the debug run.
3. **No `--cupy` flag.** The cupy kernel only covers multi-step LIF, and the
   recurrent layers are single-step. Remaining LIFs could still use it, but the
   control trained without cupy, so it is left out.
4. **Optimizer receives trainable params only.** This is identical to the
   control unless `--freeze-rec` is set.
5. **Optional extras, off by default:** `--clip-grad`, `--readout late`.
6. **Non-finite loss check.** Training stops with a message instead of
   continuing on NaNs.
7. **Extra logging fields** (above), plus `last_acc` in `checkpoint.pt`.

### From the LSM (`lsm_model.py`)

The LSM and this arm differ in more than whether recurrence is learned:

| | LSM | Recurrent SNN |
|---|---|---|
| Front end | none; each neuron reads one random pixel | trained 5-block conv stack |
| Neuron | custom: `v(1−1/τ) + I`, soft reset, heterogeneous τ and threshold | `make_lif` (SpikingJelly LIF, hard reset), same as control |
| Recurrence | 4 random inputs per neuron, fixed, weight 0.0625 | dense 3×3 conv / 512×512 linear, learned |
| Readout | 2.79M neurons avg-pooled to 512 features, linear to 11 logits | spiking FC + voting layer |
| Trainable params | 5,643 | 2.1M |
| Neuron count | 2,794,094 | 2,794,094 |

This is why `--freeze-rec` exists. It isolates **learned vs. random
recurrence** with everything else equal, which is the comparison the report's
gap statement promises. The LSM remains the classical baseline from the kickoff.

### From my initial plan

- **Separate files instead of editing shared ones.** The model lives in
  `recurrent_model.py` rather than in `models.py`, and training lives in
  `train_recurrent.py` rather than a generalized `train.py`, matching the
  one-script-per-arm layout Emma and Anna use.
- **Late readout is a wrapper.** `LateReadout` replaces the planned `readout=`
  argument on `evaluate.py`, so Emma's scorer is untouched and any arm can use it.
- **Init options are narrowed** to `orthogonal` and `zero`. The planned
  "Kaiming × 0.1" option was dropped.
- **Default is `conv5 fc`**, the two-layer main arm. The plan started with `fc`
  alone, which is still one env var away (`REC_LAYERS="fc"`).
- **Not written yet:** `probes.py` (time reversal, clockwise/counter-clockwise
  confusion, ratio of recurrent to feed-forward current), the shared spike-count
  interface, the param-matched control, and the PLIF control. The last two need
  flags in `train_control.py` (Emma's) and should be agreed with the team first.

## Checklist before the first full run

1. `python -m unittest tests.test_recurrent -v` passes in the cluster environment.
2. The debug run prints the param counts in the table above and finishes
   2 epochs. Compare time/epoch and `peak_gpu_gb` with the control.
3. Look at `rec_weight_norm` in the debug run's `metrics.jsonl`: it should
   move away from its starting value.
4. If the loss goes NaN or test accuracy collapses: `CLIP_GRAD=1.0`, then a
   smaller `REC_GAIN` (0.25), then `REC_LAYERS="fc"`, in that order.
5. The sbatch reads data from and writes runs to `/work/$USER` by default. Set
   `CONDA_ENV=<name>` to use your own conda env (otherwise it sources Emma's
   virtualenv, or `VENV=<path>`), and override `DATA_ROOT` / `OUT` to use
   someone else's dataset copy or output folder.

## Open questions for the team

- **Readout for the gap sweep.** With the mean readout, a model can answer from
  spikes summed over the whole clip, so the gap test may not separate the arms.
  Should all three arms also be trained with `--readout late`?
  (`LateReadout` works for the control and the LSM unchanged.)
- **Spike counting.** Hooks on SpikingJelly neurons will count spikes for the
  control and this arm. Inside `RecurrentLIF` the hook fires once per step
  with `[B, …]` rather than once with `[T, B, …]`, but the totals are the same.
  The LSM's spikes are computed by hand, so they need their own counter.
- **Framing in the report.** Present `--freeze-rec` as the random-recurrence
  comparison, with the LSM as the classical baseline?

## References

- Bellec et al. 2018, LSNN: learned recurrence in SNNs with BPTT.
- Yin, Corradi & Bohté 2021: spiking RNNs with adaptive neurons; orthogonal recurrent init.
- Cramer et al. 2022, Heidelberg spiking datasets: recurrent SNNs beat
  feed-forward ones with a last-step loss.
- Fang et al. 2021: PLIF (learnable τ) on DVS128 Gesture.
- SpikingJelly recurrent containers (`LinearRecurrentContainer`) concatenate
  `[x_t, y_{t-1}]` into one linear layer. That re-weights the feed-forward
  input too and only works for linear layers, which is why `RecurrentLIF` is
  custom.

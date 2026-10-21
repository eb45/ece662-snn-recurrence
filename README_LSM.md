# LSM project handoff

## Project context

This project compares three models on DVS128 Gesture:

1. **Feed-forward SNN control** — no recurrent connections; temporal behavior
   comes from neuron dynamics.
2. **LSM baseline** — fixed recurrent spiking reservoir with a trained readout.
3. **Deep recurrent SNN** — recurrent connections are learned.

The project question is whether learned recurrence improves gesture
classification and robustness to event-free time gaps compared with random
reservoir recurrence and no recurrence. The kickoff plan calls for a
same-pipeline comparison: same event bins and test split, matched neuron
counts/model capacity, same training schedule and optimizer, full-clip top-1
accuracy, accuracy after inserted gaps, and spikes per clip.

## What is in place

- `dvs_data.py` provides the shared DVS128 Gesture loading, event binning,
  person-based train/test split, and deterministic seed setup.
- `evaluate.py` provides shared full-clip top-1 evaluation. It averages model
  outputs over time and resets state between batches.
- `lsm_model.py` defines a single-liquid recurrent LIF reservoir and linear
  readout. It accepts the shared time-first input and returns per-time-step
  class scores. Its default liquid size is 2,794,094 LIF neurons, matching the
  recorded feed-forward control count.
- `train_lsm.py` reuses the shared loader/evaluator and uses the control run's
  main defaults: 64 epochs, seed 2026, batch size 8 with accumulation 2,
  T=16, Adam at 0.001, MSE loss. It saves epoch metrics, checkpoints, best
  weights, and a final summary.
- `train_lsm.sbatch` provides a launcher for the configured Duke cluster.
- `tests/test_lsm.py` contains focused model and mocked train/resume checks.

These files are **written but not yet validated**: no LSM tests, smoke run, or
training run have been executed.

## Important design caveat

The current LSM has one sampled input value per liquid neuron, four randomly
selected recurrent inputs per neuron, uniform positive recurrent weights, and
512-feature pooling before the readout. These values are implementation
choices, not prescribed by the classical LSM definition, the cited papers, or
the kickoff slides. The 512 features are not the baseline's 512 spiking hidden
neurons; they are pooled readout inputs. The LSM matches the control's LIF
count, but its trainable parameter count is much smaller. Its full-size runtime,
memory use, and task performance are unknown.

The architecture has the basic LSM form—input-driven fixed recurrent liquid
plus trained readout—but that alone does not demonstrate useful fading memory
or separation. Treat it as a prototype until reviewed and measured.

## Work remaining

### Before committing to the full run

- Review the reservoir wiring, input projection, recurrence, neuron dynamics,
  and readout design against the classical LSM goal and project fairness
  criteria. Decide whether the current design is acceptable or needs revision.
- Run the focused checks in `tests/test_lsm.py` in the target Python
  environment. They use a reduced reservoir and do not establish full-size
  feasibility or model quality.
- On the target GPU, perform a short full-size feasibility run and inspect
  memory/runtime. A reduced-neuron run is useful for debugging only; it does
  not meet the matched-neuron-count target.

### For the milestone comparison

- Train the LSM with the agreed baseline settings and full default reservoir
  size; retain its metrics and configuration.
- Compare full-clip test accuracy with the feed-forward control and recurrent
  SNN. Confirm all arms use the same data split, time bins, and evaluation
  convention.
- Record parameter count and neuron count with the accuracy results so the
  capacity differences are visible.

### Remaining project evaluation

- Implement and run the planned gap-insertion sweep without retraining; report
  accuracy by gap length and position for all arms.
- Implement a consistent spikes-per-clip measurement and report it for all
  arms.
- If required for the project's claims, evaluate fading memory and separation
  directly from reservoir states; these properties are not currently measured.

## Shared interfaces and training entry points

Use `dvs_data.py` and `evaluate.py` as the shared data/evaluation interfaces.
The control's settings and cluster conventions are documented in
`train_control.py` and `train_control.sbatch`; LSM-specific counterparts are
`train_lsm.py` and `train_lsm.sbatch`. This handoff intentionally does not
launch jobs or assume which machine/GPU environment will be used next.

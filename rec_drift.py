"""
How far W_rec moves from its initialization, and how stable it is.

For every recurrent layer:
  norm         ||W||_F (same number as rec_weight_norm in metrics.jsonl)
  drift_abs    ||W - W0||_F
  drift_rel    ||W - W0||_F / ||W0||_F           (None for zero init)
  cos_to_init  <W, W0> / (||W|| ||W0||)          (None if either is zero)
  spec_norm    largest singular value of W_rec as an operator on the layer's input.
               fc: exact. conv: power iteration on the 3x3 'same' conv over the
               layer's map (8x8 for conv5), zero padding as in the forward pass.
  radius       fc only: largest |eigenvalue| (spectral radius). For conv the
               operator is 8192-dimensional and non-normal, so power iteration
               does not reliably converge to the radius; spec_norm is an upper
               bound on it (radius <= spec_norm) and is reported for every layer.
plus "all": drift pooled over layers (sqrt of summed squares) and the max spec_norm.

Nothing here touches the global RNG, so logging these every epoch leaves the
training trajectory identical to a run without logging.

Training logs these every epoch (train_recurrent.py). For runs trained before
that, backfill end-of-training numbers. W0 is rebuilt from the seed (set_seed,
then build the model, exactly as training does), so no retraining is needed:

    python rec_drift.py /work/ah670/ece662_runs/rec_*_s2026

Each run gets <run>/drift.json (init, final = checkpoint.pt, best = best.pt) and
<run>/w0.pt if it was missing. If its final.json has no rec_init_stats /
rec_final_stats (runs trained before drift logging), they are added there too,
so collect_results.py sees the drift. Keys training already wrote are never
overwritten. For frozen runs the script checks that the rebuilt
W0 equals the trained weights bit for bit, which validates the rebuild for that
seed and config. The check allows float rounding (max |diff| <= 1e-5): the orthogonal
init goes through a QR decomposition whose last bits depend on the LAPACK build
and thread count, while a different random draw would differ by ~1e-2.
"""

import argparse
import json
import os

import torch
import torch.nn.functional as F

POWER_ITERS = 300


def rec_weights(net) -> dict:
    """{layer name: detached CPU copy of W_rec}."""
    return {name: m.rec.weight.detach().float().cpu().clone()
            for name, m in net.rec_modules().items()}


@torch.no_grad()
def linear_spectral_norm(w: torch.Tensor) -> float:
    return torch.linalg.matrix_norm(w.double(), ord=2).item()


@torch.no_grad()
def spectral_radius(w: torch.Tensor) -> float:
    return torch.linalg.eigvals(w.double()).abs().max().item()


@torch.no_grad()
def conv_spectral_norm(w: torch.Tensor, map_size: int, iters: int = POWER_ITERS) -> float:
    """Largest singular value of x -> conv2d(x, w, padding=k//2) on [C, map, map] inputs.

    Power iteration on A^T A, where A^T is conv_transpose2d with the same weight
    and padding. The estimate rises monotonically to the true value. The start
    vector comes from a private generator so the global RNG is untouched.
    """
    w = w.double()
    pad = w.shape[-1] // 2
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1, w.shape[1], map_size, map_size, generator=g, dtype=torch.float64)
    v /= v.norm()
    for _ in range(iters):
        u = F.conv2d(v, w, padding=pad)
        if u.norm() == 0:
            return 0.0
        v = F.conv_transpose2d(u, w, padding=pad)
        v /= v.norm()
    return F.conv2d(v, w, padding=pad).norm().item()


@torch.no_grad()
def conv_operator_fro2(w: torch.Tensor, map_size: int) -> float:
    """Squared Frobenius norm of the 'same' conv as a matrix on [C, map, map] inputs.

    Tap (ky, kx) contributes w[:, :, ky, kx]^2 once per output position whose
    input under that tap is inside the map (zero padding drops the rest), i.e.
    (map - |ky - pad|) * (map - |kx - pad|) times.
    """
    w = w.double()
    k, pad = w.shape[-1], w.shape[-1] // 2
    counts = torch.tensor([map_size - abs(i - pad) for i in range(k)], dtype=torch.float64)
    return ((w ** 2).sum(dim=(0, 1)) * counts[:, None] * counts[None, :]).sum().item()


@torch.no_grad()
def drift_stats(weights: dict, w0: dict, map_sizes: dict) -> dict:
    out, d2_sum, n02_sum = {}, 0.0, 0.0
    for name, w in weights.items():
        w, a = w.double(), w0[name].double()
        n, n0, d = w.norm().item(), a.norm().item(), (w - a).norm().item()
        row = {
            "norm": n,
            "drift_abs": d,
            "drift_rel": d / n0 if n0 > 0 else None,
            "cos_to_init": (w * a).sum().item() / (n * n0) if n > 0 and n0 > 0 else None,
        }
        if w.dim() == 2:
            row["spec_norm"] = linear_spectral_norm(w)
            row["radius"] = spectral_radius(w)
            fro2 = n * n
        else:
            row["spec_norm"] = conv_spectral_norm(w, map_sizes[name])
            fro2 = conv_operator_fro2(w, map_sizes[name])  # whole operator, not the kernel
        # Stable rank ||A||_F^2 / ||A||_2^2 of the recurrent operator, in [1, dim]: how many
        # directions carry its energy. Orthogonal fc init gives the full 512; a few dominant
        # modes give a small number. For conv5 the operator acts on 128 x 8 x 8 = 8192 dims.
        row["stable_rank"] = fro2 / row["spec_norm"] ** 2 if row["spec_norm"] > 0 else None
        out[name] = row
        d2_sum += d * d
        n02_sum += n0 * n0
    out["all"] = {
        "drift_abs": d2_sum ** 0.5,
        "drift_rel": (d2_sum / n02_sum) ** 0.5 if n02_sum > 0 else None,
        "spec_norm_max": max(r["spec_norm"] for r in out.values()),
    }
    return out


def net_drift_stats(net, w0: dict) -> dict:
    return drift_stats(rec_weights(net), w0, net.rec_map_sizes())


def rebuild_w0(args: dict) -> dict:
    """W_rec at initialization for a run, from the args in its final.json."""
    from dvs_data import set_seed
    from recurrent_model import build_from_args

    set_seed(args["seed"])
    return rec_weights(build_from_args(args))


def fmt_stats(stats: dict) -> str:
    """One-line summary for the training log."""
    parts = []
    for name, r in stats.items():
        if name == "all":
            continue
        rel = "-" if r["drift_rel"] is None else f"{r['drift_rel']:.3f}"
        s = f"{name}: drift {rel} sn {r['spec_norm']:.3f}"
        if "radius" in r:
            s += f" rho {r['radius']:.3f}"
        if r.get("stable_rank") is not None:
            s += f" srank {r['stable_rank']:.1f}"
        parts.append(s)
    return "  ".join(parts)


MATCH_ATOL = 1e-5  # entries are ~0.02; a different random draw differs by ~0.02, QR rounding by ~1e-7


def max_abs_diff(a: dict, b: dict) -> float:
    return max((a[k].float() - b[k].float()).abs().max().item() for k in a)


def backfill(run_dir: str) -> dict:
    from recurrent_model import build_from_args

    with open(os.path.join(run_dir, "final.json")) as f:
        args = json.load(f)["args"]
    w0 = rebuild_w0(args)
    net = build_from_args(args)
    sizes = net.rec_map_sizes()
    res = {"init": drift_stats(w0, w0, sizes)}

    w0_path = os.path.join(run_dir, "w0.pt")
    if os.path.exists(w0_path):
        saved = torch.load(w0_path, map_location="cpu")
        res["w0_saved_max_diff"] = max_abs_diff(saved, w0)
        res["w0_matches_saved"] = res["w0_saved_max_diff"] <= MATCH_ATOL
    else:
        torch.save(w0, w0_path)

    for key, fname in (("final", "checkpoint.pt"), ("best", "best.pt")):
        path = os.path.join(run_dir, fname)
        if not os.path.exists(path):
            continue
        state = torch.load(path, map_location="cpu")
        if "model" in state and "opt" in state:
            state = state["model"]
        net.load_state_dict(state)
        res[key] = drift_stats(rec_weights(net), w0, sizes)
        if key == "final" and args["freeze_rec"]:
            res["frozen_max_diff"] = max_abs_diff(rec_weights(net), w0)
            res["frozen_check"] = res["frozen_max_diff"] <= MATCH_ATOL

    with open(os.path.join(run_dir, "drift.json"), "w") as f:
        json.dump(res, f, indent=2)
    res["final_json_updated"] = add_stats_to_final_json(run_dir, res)
    return res


def add_stats_to_final_json(run_dir: str, res: dict) -> bool:
    """Give an old run's final.json the rec_init_stats / rec_final_stats keys that
    training now writes, so collect_results.py picks up its drift.

    Only adds missing keys and never overwrites ones training wrote. The final
    stats come from checkpoint.pt (the last epoch), matching what training logs.
    Returns True if final.json was changed.
    """
    if "final" not in res:
        return False
    path = os.path.join(run_dir, "final.json")
    with open(path) as f:
        final = json.load(f)
    added = False
    for key, stats in (("rec_init_stats", res["init"]), ("rec_final_stats", res["final"])):
        if key not in final:
            final[key] = stats
            added = True
    if added:
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(final, f, indent=2)
        os.replace(tmp, path)
    return added


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("runs", nargs="+", help="Run folders containing final.json")
    for run in p.parse_args().runs:
        if not os.path.exists(os.path.join(run, "final.json")):
            print(f"skip {run}: no final.json")
            continue
        res = backfill(run)
        name = os.path.basename(os.path.normpath(run))
        print(f"{name}\n  init   {fmt_stats(res['init'])}")
        if "final" in res:
            print(f"  final  {fmt_stats(res['final'])}")
        if "frozen_check" in res:
            print(f"  frozen check (rebuilt W0 == trained W): "
                  f"{'OK' if res['frozen_check'] else 'MISMATCH'} (max |diff| {res['frozen_max_diff']:.2e})")
        if "w0_matches_saved" in res:
            print(f"  rebuilt W0 == saved w0.pt: {'OK' if res['w0_matches_saved'] else 'MISMATCH'}"
                  f" (max |diff| {res['w0_saved_max_diff']:.2e})")
        if res["final_json_updated"]:
            print("  added rec_init_stats / rec_final_stats to final.json")


if __name__ == "__main__":
    main()

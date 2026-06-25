#!/usr/bin/env python3
"""
mesh_metrics.py

Compute 3D surface distances between a predicted mesh and GT mesh
already expressed in the same frame.

3D metrics:
- Sample N points on each mesh surface
- Nearest-neighbor distances pred->gt and gt->pred
- Stats + histograms (fraction-of-points, log-x)

Outputs (if --out_dir is given):
- pred_to_gt.npy, gt_to_pred.npy
- hist_*.png
- stats.json
"""

import argparse
from pathlib import Path
import json
import os
import tempfile
import numpy as np

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt


# ============================================================
# Stats + plotting
# ============================================================
def stats(d: np.ndarray) -> dict:
    """Compute robust descriptive statistics for a distance array."""
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    if d.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "median": float("nan"),
                "p95": float("nan"), "max": float("nan")}
    return {
        "mean":   float(np.mean(d)),
        "std":    float(np.std(d)),
        "median": float(np.median(d)),
        "p95":    float(np.quantile(d, 0.95)),
        "max":    float(np.max(d)),
    }


def print_stats(title: str, s: dict):
    """Print a metric dictionary using scientific notation."""
    print(f"\n{title}")
    for k, v in s.items():
        print(f"  {k:>8s}: {v:.6e}")


def plot_histogram_fraction(
    d: np.ndarray,
    title: str,
    xlabel: str,
    save_path: Path | None = None,
    n_bins: int = 100,
):
    """Plot a log-x histogram where bar heights are fractions of sampled points."""
    d = np.asarray(d)
    d = d[np.isfinite(d)]
    d = d[d > 0]
    if d.size == 0:
        print(f"[WARN] No valid values for histogram: {title}")
        return

    bins    = np.logspace(np.log10(d.min()), np.log10(d.max()), n_bins)
    weights = np.ones_like(d) / len(d)

    plt.figure(figsize=(7, 5))
    plt.hist(d, bins=bins, weights=weights, alpha=0.75)
    plt.axvline(np.median(d),         color="black", linestyle="--", label="median")
    plt.axvline(np.quantile(d, 0.95), color="red",   linestyle="--", label="p95")
    plt.xscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("Fraction of points")
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    plt.show()
    plt.close()


def run_mesh_metrics(args) -> None:
    """Run the full mesh-to-mesh evaluation and optionally save all outputs."""
    from metrics_utils import (
        directed_hausdorff,
        directed_hausdorff_p95,
        load_mesh,
        nn_distances,
        sample_surface_points,
        symmetric_chamfer,
        symmetric_hausdorff,
        symmetric_hausdorff_p95,
    )

    if args.out_dir is not None:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load meshes ───────────────────────────────────────────
    print(f"Loading pred mesh: {args.pred_mesh}")
    pred = load_mesh(args.pred_mesh)
    print(f"Loading GT mesh:   {args.gt_mesh}")
    gt   = load_mesh(args.gt_mesh)

    print("Assuming pred and GT are already in the same frame.")

    # ── Sample surfaces ───────────────────────────────────────
    gt_seed = None if args.seed == -1 else int(args.seed)
    pred_seed = None if args.seed == -1 else int(args.seed) + 1

    G = sample_surface_points(gt, args.n, seed=gt_seed)
    P = sample_surface_points(pred, args.n, seed=pred_seed)

    # ── 3D Chamfer / Hausdorff ────────────────────────────────
    dP = nn_distances(P, G)   # pred → gt
    dG = nn_distances(G, P)   # gt → pred

    sP = stats(dP)
    sG = stats(dG)
    chamfer_sym = symmetric_chamfer(dP, dG)
    hausdorff_pred_to_gt = directed_hausdorff(dP)
    hausdorff_gt_to_pred = directed_hausdorff(dG)
    hausdorff_sym = symmetric_hausdorff(dP, dG)
    hausdorff_p95_pred_to_gt = directed_hausdorff_p95(dP)
    hausdorff_p95_gt_to_pred = directed_hausdorff_p95(dG)
    hausdorff_p95_sym = symmetric_hausdorff_p95(dP, dG)

    print_stats("Pred → GT statistics (3D)", sP)
    print_stats("GT → Pred statistics (3D)", sG)
    print(f"\nSymmetric Chamfer (mean, 3D): {chamfer_sym:.6e}")
    print(f"Directed Hausdorff pred → gt (3D): {hausdorff_pred_to_gt:.6e}")
    print(f"Directed Hausdorff gt → pred (3D): {hausdorff_gt_to_pred:.6e}")
    print(f"Symmetric Hausdorff (3D): {hausdorff_sym:.6e}")
    print(f"Symmetric Hausdorff-p95 (3D): {hausdorff_p95_sym:.6e}")

    plot_histogram_fraction(
        dP, "Distance error distribution (pred → gt)",
        xlabel="Distance (mesh units)",
        save_path=(args.out_dir / "hist_pred_to_gt.png" if args.out_dir else None),
    )
    plot_histogram_fraction(
        dG, "Distance error distribution (gt → pred)",
        xlabel="Distance (mesh units)",
        save_path=(args.out_dir / "hist_gt_to_pred.png" if args.out_dir else None),
    )

    # ── Save outputs ──────────────────────────────────────────
    if args.out_dir is not None:
        np.save(args.out_dir / "pred_to_gt.npy", dP)
        np.save(args.out_dir / "gt_to_pred.npy", dG)
        payload = {
            "pred_to_gt_3d":        sP,
            "gt_to_pred_3d":        sG,
            "symmetric_chamfer_3d": chamfer_sym,
            "directed_hausdorff_pred_to_gt_3d": hausdorff_pred_to_gt,
            "directed_hausdorff_gt_to_pred_3d": hausdorff_gt_to_pred,
            "hausdorff_3d": hausdorff_sym,
            "directed_hausdorff_p95_pred_to_gt_3d": hausdorff_p95_pred_to_gt,
            "directed_hausdorff_p95_gt_to_pred_3d": hausdorff_p95_gt_to_pred,
            "hausdorff_p95_3d": hausdorff_p95_sym,
            "n":    args.n,
            "seed": args.seed,
        }
        with open(args.out_dir / "stats.json", "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nResults saved to: {args.out_dir}")


# ============================================================
# Main
# ============================================================
def main():
    """CLI entry point for one predicted mesh against one GT mesh."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred_mesh",    type=Path, required=True)
    ap.add_argument("--gt_mesh",      type=Path, required=True)
    ap.add_argument("--n",            type=int,  default=100_000)
    ap.add_argument("--seed",         type=int,  default=42)
    ap.add_argument("--out_dir",      type=Path, default=None,
                    help="If provided, save .npy arrays, stats.json and histogram PNGs")
    args = ap.parse_args()

    run_mesh_metrics(args)


if __name__ == "__main__":
    main()

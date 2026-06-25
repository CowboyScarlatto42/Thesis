#!/usr/bin/env python3
"""
Plot NeuS mesh metrics over iterations for multiple experiments.

Required files in each metrics directory:
  - pred_to_gt_stats.npy, shape (N, 5)
  - gt_to_pred_stats.npy, shape (N, 5)
  - symchamfer.npy, shape (N,) or (N, 1)
  - hausdorff_distance.npy, shape (N,) or (N, 1)
  - hausdorff_p95_distance.npy, shape (N,) or (N, 1)
  - iterations.npy, shape (N,)

The stats columns must be:
  mean, std, median, p95, max

Always produces:
  - pred_to_gt_mean.png, pred_to_gt_std.png, pred_to_gt_median.png
  - pred_to_gt_p95.png, pred_to_gt_max.png
  - gt_to_pred_mean.png, gt_to_pred_std.png, gt_to_pred_median.png
  - gt_to_pred_p95.png, gt_to_pred_max.png
  - symmetric_chamfer.png
  - hausdorff_distance.png
  - hausdorff_p95_distance.png
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ:
    mpl_cache_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_cache_dir)

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np


STAT_KEYS = ["mean", "std", "median", "p95", "max"]
STAT_LABELS = {
    "mean": "Mean distance",
    "std": "Std distance",
    "median": "Median distance",
    "p95": "95th percentile distance",
    "max": "Max distance",
}

BASE_PLOTS = (
    *[
        (f"pred_to_gt_{stat}.png", "pred_to_gt", stat, f"Pred -> GT - {STAT_LABELS[stat]}")
        for stat in STAT_KEYS
    ],
    *[
        (f"gt_to_pred_{stat}.png", "gt_to_pred", stat, f"GT -> Pred - {STAT_LABELS[stat]}")
        for stat in STAT_KEYS
    ],
    ("symmetric_chamfer.png", "symchamfer", None, "Symmetric Chamfer distance"),
    ("hausdorff_distance.png", "hausdorff_distance", None, "Hausdorff distance"),
    ("hausdorff_p95_distance.png", "hausdorff_p95_distance", None, "Hausdorff distance p95"),
)

EXTRA_SERIES_PLOTS = (
    (
        "directed_hausdorff_pred_to_gt.png",
        "directed_hausdorff_pred_to_gt",
        "Directed Hausdorff pred -> GT",
    ),
    (
        "directed_hausdorff_gt_to_pred.png",
        "directed_hausdorff_gt_to_pred",
        "Directed Hausdorff GT -> pred",
    ),
    (
        "directed_hausdorff_p95_pred_to_gt.png",
        "directed_hausdorff_p95_pred_to_gt",
        "Directed Hausdorff-p95 pred -> GT",
    ),
    (
        "directed_hausdorff_p95_gt_to_pred.png",
        "directed_hausdorff_p95_gt_to_pred",
        "Directed Hausdorff-p95 GT -> pred",
    ),
)


def parse_experiment_arg(value: str) -> dict:
    """Parse one LABEL=METRICS_DIR command-line experiment specification."""
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Experiment must use LABEL=METRICS_DIR format, "
            f"got: {value}"
        )
    label, metrics_dir = value.split("=", 1)
    label = label.strip()
    metrics_dir = metrics_dir.strip()
    if not label or not metrics_dir:
        raise argparse.ArgumentTypeError(
            "Experiment label and metrics directory must be non-empty."
        )
    return {"label": label, "metrics_dir": Path(metrics_dir)}


def load_experiments_json(path: Path) -> list[dict]:
    """Load experiment labels and metric directories from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict):
        payload = payload.get("experiments")

    if not isinstance(payload, list):
        raise ValueError("Experiments JSON must be a list or a dict with key 'experiments'.")

    experiments = []
    for item in payload:
        if not isinstance(item, dict) or "label" not in item or "metrics_dir" not in item:
            raise ValueError("Each experiment must contain 'label' and 'metrics_dir'.")
        experiments.append({
            "label": str(item["label"]),
            "metrics_dir": Path(item["metrics_dir"]),
        })
    return experiments


def load_required_array(metrics_dir: Path, filename: str) -> np.ndarray:
    """Load a required .npy file from one experiment directory."""
    path = metrics_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    return np.load(path)


def as_1d(arr: np.ndarray, name: str, metrics_dir: Path) -> np.ndarray:
    """Accept either (N,) or (N, 1) arrays and return a flat vector."""
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    raise ValueError(f"{name} in {metrics_dir} must have shape (N,) or (N, 1), got {arr.shape}")


def validate_stats(name: str, arr: np.ndarray, metrics_dir: Path) -> None:
    """Check that a stats array contains at least the expected five columns."""
    if arr.ndim != 2 or arr.shape[1] < len(STAT_KEYS):
        raise ValueError(f"{name} in {metrics_dir} must have shape (N, 5), got {arr.shape}")


def validate_lengths(data: dict, metrics_dir: Path) -> None:
    """Ensure every metric series has the same number of iterations."""
    n = data["iterations"].shape[0]
    for key, arr in data.items():
        if key == "metrics_dir":
            continue
        if arr.shape[0] != n:
            raise ValueError(
                f"Inconsistent length for {key} in {metrics_dir}: "
                f"{arr.shape[0]} values, expected {n}"
            )


def load_experiment(metrics_dir: Path) -> dict:
    """Load and validate all required metric arrays for one experiment."""
    data = {
        "metrics_dir": metrics_dir,
        "pred_to_gt": load_required_array(metrics_dir, "pred_to_gt_stats.npy"),
        "gt_to_pred": load_required_array(metrics_dir, "gt_to_pred_stats.npy"),
        "symchamfer": as_1d(load_required_array(metrics_dir, "symchamfer.npy"), "symchamfer.npy", metrics_dir),
        "hausdorff_distance": as_1d(
            load_required_array(metrics_dir, "hausdorff_distance.npy"),
            "hausdorff_distance.npy",
            metrics_dir,
        ),
        "hausdorff_p95_distance": as_1d(
            load_required_array(metrics_dir, "hausdorff_p95_distance.npy"),
            "hausdorff_p95_distance.npy",
            metrics_dir,
        ),
        "iterations": as_1d(load_required_array(metrics_dir, "iterations.npy"), "iterations.npy", metrics_dir),
    }

    validate_stats("pred_to_gt_stats.npy", data["pred_to_gt"], metrics_dir)
    validate_stats("gt_to_pred_stats.npy", data["gt_to_pred"], metrics_dir)

    validate_lengths(data, metrics_dir)
    return data


def get_series(data: dict, array_key: str, stat: str | None) -> np.ndarray:
    """Extract either a scalar metric series or one stats column."""
    arr = data[array_key]
    if stat is None:
        return as_1d(arr, array_key, data["metrics_dir"])
    return arr[:, STAT_KEYS.index(stat)]


def make_plot(
    filename: str,
    title: str,
    values_for_experiment,
    loaded: list[dict],
    experiments_cfg: list[dict],
    out_dir: Path,
    args: argparse.Namespace,
) -> None:
    """Create one comparison plot across all loaded experiments."""
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))

    for exp_data, exp_cfg in zip(loaded, experiments_cfg):
        iters = exp_data["iterations"] / 1_000.0
        values = values_for_experiment(exp_data)

        ax.plot(
            iters,
            values,
            label=exp_cfg["label"],
            linewidth=args.line_width,
            marker=args.marker,
            markersize=args.marker_size,
        )

    ax.set_xlabel("Iterations (x10^3)", fontsize=11)
    ax.set_ylabel(args.ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax.grid(True, alpha=args.grid_alpha)
    ax.legend(fontsize=9, loc="best")

    plt.tight_layout()
    save_path = out_dir / filename
    plt.savefig(save_path, dpi=args.dpi)
    if args.show:
        plt.show()
    plt.close()

    print(f"  Saved: {save_path}")


def format_metric(value: float) -> str:
    """Format a floating metric for terminal tables."""
    return f"{float(value):.6e}"


def format_int(value: float) -> str:
    """Format an iteration value as an integer string."""
    return str(int(round(float(value))))


def find_iteration_index(iterations: np.ndarray, target_iter: int) -> tuple[int, bool]:
    """Return the exact target iteration index, or the nearest available one."""
    iterations = np.asarray(iterations)
    matches = np.where(iterations == target_iter)[0]
    if matches.size > 0:
        return int(matches[0]), True
    return int(np.argmin(np.abs(iterations - target_iter))), False


def print_metrics_table(
    loaded: list[dict],
    experiments_cfg: list[dict],
    target_iter: int,
) -> None:
    """Print a compact terminal table at the requested training iteration."""
    rows = []
    used_nearest = False

    for exp_data, exp_cfg in zip(loaded, experiments_cfg):
        idx, exact = find_iteration_index(exp_data["iterations"], target_iter)
        used_nearest = used_nearest or not exact
        iter_label = format_int(exp_data["iterations"][idx])
        if not exact:
            iter_label += "*"

        # include all statistics for pred->gt and gt->pred
        row = {
            "case": exp_cfg["label"],
            "iter": iter_label,
            "chamfer": format_metric(exp_data["symchamfer"][idx]),
            "hausdorff": format_metric(exp_data["hausdorff_distance"][idx]),
            "hausdorff_p95": format_metric(exp_data["hausdorff_p95_distance"][idx]),
        }
        for stat in STAT_KEYS:
            row[f"p2g_{stat}"] = format_metric(exp_data["pred_to_gt"][idx, STAT_KEYS.index(stat)])
            row[f"g2p_{stat}"] = format_metric(exp_data["gt_to_pred"][idx, STAT_KEYS.index(stat)])
        rows.append(row)

    # build columns to show all stats for pred->gt and gt->pred
    columns = [
        ("case", "Case"),
        ("iter", "Iter"),
    ]
    for prefix, title_prefix in (("p2g", "P->G"), ("g2p", "G->P")):
        for stat in STAT_KEYS:
            columns.append((f"{prefix}_{stat}", f"{title_prefix} {stat}"))
    columns += [
        ("chamfer", "Chamfer"),
        ("hausdorff", "Hausdorff"),
        ("hausdorff_p95", "Hausdorff p95"),
    ]

    widths = {}
    for key, title in columns:
        widths[key] = max(len(title), *(len(row[key]) for row in rows))

    print(f"\nMetrics table at iter={target_iter}")
    header = " | ".join(title.ljust(widths[key]) for key, title in columns)
    sep = "-+-".join("-" * widths[key] for key, _ in columns)
    print(header)
    print(sep)
    for row in rows:
        print(" | ".join(row[key].ljust(widths[key]) for key, _ in columns))

    if used_nearest:
        print("\n* exact iteration not found; nearest available iteration used.")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for multi-experiment metric plotting."""
    parser = argparse.ArgumentParser(
        description="Plot NeuS mesh metrics over iterations for multiple experiments."
    )
    parser.add_argument(
        "--experiment",
        action="append",
        type=parse_experiment_arg,
        default=[],
        help="Experiment in LABEL=METRICS_DIR format. Can be repeated.",
    )
    parser.add_argument(
        "--experiments_json",
        type=Path,
        default=None,
        help="Optional JSON list with objects containing label and metrics_dir.",
    )
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--fig_width", type=float, default=9.0)
    parser.add_argument("--fig_height", type=float, default=5.0)
    parser.add_argument("--line_width", type=float, default=1.8)
    parser.add_argument("--marker", default="o")
    parser.add_argument("--marker_size", type=float, default=3.0)
    parser.add_argument("--grid_alpha", type=float, default=0.3)
    parser.add_argument("--ylabel", default="Distance (m)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--table_iter", type=int, default=300000)
    parser.add_argument(
        "--include_directed_hausdorff",
        action="store_true",
        help="Also plot directed Hausdorff arrays if all required files exist.",
    )
    return parser.parse_args()


def main() -> None:
    """Load experiments, generate plots, and print the selected-iteration table."""
    args = parse_args()
    experiments = list(args.experiment)

    if args.experiments_json is not None:
        experiments.extend(load_experiments_json(args.experiments_json))

    if not experiments:
        raise SystemExit("Provide at least one --experiment or --experiments_json.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading experiments...")
    loaded = []
    for cfg in experiments:
        print(f"  {cfg['label']} <- {cfg['metrics_dir']}")
        loaded.append(load_experiment(cfg["metrics_dir"]))

    print(f"\nGenerating {len(BASE_PLOTS)} plots...\n")
    for filename, array_key, stat, title in BASE_PLOTS:
        make_plot(
            filename=filename,
            title=title,
            values_for_experiment=lambda data, key=array_key, stat=stat: get_series(data, key, stat),
            loaded=loaded,
            experiments_cfg=experiments,
            out_dir=args.out_dir,
            args=args,
        )

    if args.include_directed_hausdorff:
        for filename, key, title in EXTRA_SERIES_PLOTS:
            for exp_data in loaded:
                metrics_dir = exp_data["metrics_dir"]
                exp_data[key] = as_1d(
                    load_required_array(metrics_dir, f"{key}.npy"),
                    f"{key}.npy",
                    metrics_dir,
                )
            validate_lengths({k: v for k, v in loaded[0].items() if k in ("iterations", key)}, loaded[0]["metrics_dir"])
            make_plot(
                filename=filename,
                title=title,
                values_for_experiment=lambda data, key=key: data[key],
                loaded=loaded,
                experiments_cfg=experiments,
                out_dir=args.out_dir,
                args=args,
            )

    total_plots = len(BASE_PLOTS) + (len(EXTRA_SERIES_PLOTS) if args.include_directed_hausdorff else 0)
    print(f"\nSaved {total_plots} plots to: {args.out_dir}")
    print_metrics_table(loaded, experiments, args.table_iter)


if __name__ == "__main__":
    main()

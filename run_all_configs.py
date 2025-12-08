#!/usr/bin/env python3
"""Orchestrate NeuS experiments across multiple view-count configurations.

This script keeps NeuS untouched and simply coordinates training, mesh
extraction, and evaluation via subprocess calls to ``exp_runner.py``
and an external Chamfer-distance calculator.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib
import pandas as pd

matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt  # noqa: E402  (after backend selection)


###############################################################################
# Configuration section (edit paths here as needed)
###############################################################################

# This script no longer assumes absolute paths; they are constructed at runtime
# based on CLI arguments.


@dataclass
class ExperimentConfig:
    name: str
    case: str
    conf_rel: Path
    n_views: int
    n_iters: int
    extra_args: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "ExperimentConfig":
        return cls(
            name=str(data["name"]),
            case=str(data["case"]),
            conf_rel=Path(data["conf_rel"]),
            n_views=int(data.get("n_views", 0)),
            n_iters=int(data.get("n_iters", 0)),
            extra_args=list(data.get("extra_args", [])),
        )


CONFIGS: List[ExperimentConfig] = [
    ExperimentConfig.from_dict(
        {
            "name": "neus_1view",
            "case": "hst_neus_1views",
            "conf_rel": "hst_spe3r_case.conf",  
            "n_views": 1,
            "n_iters": 10000,
        }
    ),
    ExperimentConfig.from_dict(
        {
            "name": "neus_4views",
            "case": "hst_neus_4views",
            "conf_rel": "hst_spe3r_case.conf",  
            "n_views": 4,
            "n_iters": 10000,
        }
    ),
    ExperimentConfig.from_dict(
        {
            "name": "neus_8views",
            "case": "hst_neus_8views",
            "conf_rel": "hst_spe3r_case.conf",  
            "n_views": 8,
            "n_iters": 10000,
        }
    ),
]




###############################################################################
# Utility helpers
###############################################################################

def _ensure_exists(path: Path, descriptor: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{descriptor} not found: {path}")


def run_subprocess(cmd: Sequence[str], cwd: Optional[Path] = None) -> float:
    """Run a subprocess command and return elapsed wall-clock time in seconds."""
    print(f"\n[run_subprocess] Executing: {' '.join(cmd)}")
    start = time.time()
    result = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)
    end = time.time()
    elapsed = end - start
    print(f"[run_subprocess] Completed in {elapsed:.2f} s")
    return elapsed


###############################################################################
# Experiment step helpers
###############################################################################

def run_train(exp_runner: Path, conf_path: Path, config: ExperimentConfig, cwd: Path) -> float:
    _ensure_exists(exp_runner, "exp_runner.py")
    _ensure_exists(conf_path, f"config file for {config.name}")

    cmd = [
        sys.executable,
        str(exp_runner),
        "--mode",
        "train",
        "--conf",
        str(conf_path),
        "--case",
        config.case,
    ] + config.extra_args

    print(f"\n=== Training: {config.name} ({config.case}) ===")
    elapsed = run_subprocess(cmd, cwd=cwd)
    return elapsed


def run_inference(exp_runner: Path, conf_path: Path, config: ExperimentConfig, cwd: Path) -> float:
    _ensure_exists(exp_runner, "exp_runner.py")

    cmd = [
        sys.executable,
        str(exp_runner),
        "--mode",
        "validate_mesh",  # TODO: adjust mode if a different inference entrypoint is required
        "--conf",
        str(conf_path),
        "--case",
        config.case,
        "--is_continue",
    ] + config.extra_args

    print(f"\n=== Inference (mesh extraction): {config.name} ===")
    elapsed = run_subprocess(cmd, cwd=cwd)
    return elapsed


###############################################################################
# Evaluation helpers
###############################################################################

def get_pred_mesh_path(exp_root: Path, config: ExperimentConfig) -> Path:
    meshes_dir = exp_root / config.case / "wmask" / "meshes"
    if not meshes_dir.exists():
        raise FileNotFoundError(f"Meshes directory not found for {config.name}: {meshes_dir}")

    pred_mesh = meshes_dir / "pred.ply"
    if pred_mesh.exists():
        print(f"[get_pred_mesh_path] Using explicit mesh: {pred_mesh}")
        return pred_mesh

    ply_files = sorted(meshes_dir.glob("*.ply"))
    if not ply_files:
        raise FileNotFoundError(f"No .ply meshes found in {meshes_dir}")

    latest = max(ply_files, key=lambda p: p.stat().st_mtime)
    print(f"[get_pred_mesh_path] Using latest mesh: {latest}")
    return latest


def compute_chamfer(chamfer_script: Path, exp_root: Path, gt_mesh: Path, config: ExperimentConfig) -> float:
    _ensure_exists(chamfer_script, "Chamfer calculator script")


    pred_mesh = get_pred_mesh_path(exp_root, config)
    if not gt_mesh.exists():
        raise FileNotFoundError(f"Ground-truth mesh not found: {gt_mesh}")

    cmd = [
        sys.executable,
        str(chamfer_script),
        "--pred_mesh",
        str(pred_mesh),
        "--gt_mesh",
        str(gt_mesh),
    ]

    print(f"\n=== Chamfer distance: {config.name} ===")
    print(f"Pred mesh: {pred_mesh}")
    print(f"GT mesh:   {gt_mesh}")

    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stdout = completed.stdout.strip()
    print(stdout)

    # Try to parse JSON first, otherwise look for the last float in stdout.
    chamfer_value: Optional[float] = None
    try:
        data = json.loads(stdout.splitlines()[-1])
        if isinstance(data, dict) and "cd_l2_sq" in data:
            chamfer_value = float(data["cd_l2_sq"])
    except json.JSONDecodeError:
        pass

    if chamfer_value is None:
        for line in stdout.splitlines():
            if "cd_l2_sq" in line:
                maybe_value = line.split(":")[-1].strip()
                if _looks_like_float(maybe_value):
                    chamfer_value = float(maybe_value)
                    break
        else:
            tokens = stdout.split()
            float_tokens = [t for t in tokens if _looks_like_float(t)]
            if not float_tokens:
                raise ValueError("Unable to parse Chamfer distance from chamfer script output")
            chamfer_value = float(float_tokens[-1])

    print(f"[compute_chamfer] Chamfer (L2^2): {chamfer_value:.6e}")
    return chamfer_value


def _looks_like_float(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


###############################################################################
# Optional logging utilities
###############################################################################

def load_convergence_log(exp_root: Path, config: ExperimentConfig) -> Optional[pd.DataFrame]:
    log_path = exp_root / config.case / "logs" / "train_log.csv"
    if not log_path.exists():
        print(f"[load_convergence_log] No log file for {config.name}: {log_path}")
        return None
    try:
        df = pd.read_csv(log_path)
        required_cols = {"iter", "loss"}
        if not required_cols.issubset(df.columns):
            print(f"[load_convergence_log] Missing required columns in {log_path}")
            return None
        df = df.sort_values("iter")
        print(f"[load_convergence_log] Loaded {len(df)} rows from {log_path}")
        return df
    except Exception as exc:  # pragma: no cover - defensive logging
        print(f"[load_convergence_log] Failed to read {log_path}: {exc}")
        return None


###############################################################################
# Plotting helpers
###############################################################################

def plot_summary(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(df["n_views"], df["chamfer"], marker="o")
    plt.title("Chamfer distance vs number of views")
    plt.xlabel("Number of views")
    plt.ylabel("Chamfer distance (L2^2)")
    plt.grid(True, alpha=0.3)
    chamfer_plot = output_dir / "chamfer_vs_views.png"
    plt.savefig(chamfer_plot, bbox_inches="tight")
    plt.close()
    print(f"[plot_summary] Saved {chamfer_plot}")

    plt.figure(figsize=(6, 4))
    plt.plot(df["n_views"], df["train_time_s"], marker="o", color="tab:orange")
    plt.title("Training time vs number of views")
    plt.xlabel("Number of views")
    plt.ylabel("Training time (s)")
    plt.grid(True, alpha=0.3)
    train_time_plot = output_dir / "train_time_vs_views.png"
    plt.savefig(train_time_plot, bbox_inches="tight")
    plt.close()
    print(f"[plot_summary] Saved {train_time_plot}")


def plot_convergence(logs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    logs = {k: v for k, v in logs.items() if v is not None}
    if not logs:
        print("[plot_convergence] No convergence logs available; skipping plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    for name, df in logs.items():
        plt.plot(df["iter"], df["loss"], label=name)
    plt.title("Training loss convergence")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    convergence_plot = output_dir / "loss_convergence.png"
    plt.savefig(convergence_plot, bbox_inches="tight")
    plt.close()
    print(f"[plot_convergence] Saved {convergence_plot}")


###############################################################################
# Main entry point
###############################################################################

def main(argv: Optional[Iterable[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run NeuS experiments across configurations.")
    parser.add_argument("--repo-root", type=str, default=".", help="Root directory of the NeuS project.")
    parser.add_argument("--conf-dir", type=str, default="confs", help="Directory with NeuS .conf files (relative to repo root).")
    parser.add_argument("--exp-dir", type=str, default="exp", help="Directory where NeuS stores experiment outputs (relative to repo root).")
    parser.add_argument("--gt-mesh", type=str, required=True, help="Ground-truth mesh path (.ply). Relative paths are resolved against repo root.")
    parser.add_argument("--results-dir", type=str, default="results_wrapper", help="Directory (relative to repo root) to store wrapper outputs.")
    parser.add_argument("--skip-train", action="store_true", help="Skip the training phase.")
    parser.add_argument("--skip-inference", action="store_true", help="Skip the inference / mesh extraction phase.")
    parser.add_argument("--skip-chamfer", action="store_true", help="Skip Chamfer distance computation.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = Path(args.repo_root).resolve()
    script_dir = Path(__file__).resolve().parent
    conf_dir = (repo_root / args.conf_dir).resolve()
    exp_root = (repo_root / args.exp_dir).resolve()
    gt_mesh_arg = Path(args.gt_mesh)
    gt_mesh = gt_mesh_arg if gt_mesh_arg.is_absolute() else (repo_root / gt_mesh_arg).resolve()
    results_dir = (repo_root / args.results_dir).resolve()

    exp_runner = repo_root / "exp_runner.py"
    chamfer_script = script_dir / "chamfer_calculator.py"

    results_dir.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, object]] = []
    convergence_logs: Dict[str, Optional[pd.DataFrame]] = {}

    try:
        for config in CONFIGS:
            print("\n############################################################")
            print(f"Starting experiment: {config.name} ({config.case})")
            print("############################################################")

            conf_path = (conf_dir / config.conf_rel).resolve()

            if args.skip_train:
                train_time = float("nan")
            else:
                train_time = run_train(exp_runner, conf_path, config, cwd=repo_root)

            if args.skip_inference:
                inference_time = float("nan")
            else:
                inference_time = run_inference(exp_runner, conf_path, config, cwd=repo_root)

            if args.skip_chamfer:
                chamfer_value = float("nan")
                chamfer_time = float("nan")
            else:
                chamfer_start = time.time()
                chamfer_value = compute_chamfer(chamfer_script, exp_root, gt_mesh, config)
                chamfer_time = time.time() - chamfer_start

            convergence_logs[config.name] = load_convergence_log(exp_root, config)

            results.append(
                {
                    "name": config.name,
                    "case": config.case,
                    "n_views": config.n_views,
                    "n_iters": config.n_iters,
                    "train_time_s": train_time,
                    "inference_time_s": inference_time,
                    "chamfer": chamfer_value,
                    "chamfer_time_s": chamfer_time,
                }
            )

    except subprocess.CalledProcessError as exc:
        print("\n[main] Subprocess failed with non-zero exit status.")
        print(f"Command: {' '.join(exc.cmd) if hasattr(exc, 'cmd') else exc}")
        print(f"Return code: {exc.returncode}")
        if exc.stdout:
            print("--- stdout ---")
            print(exc.stdout)
        if exc.stderr:
            print("--- stderr ---")
            print(exc.stderr)
        raise
    except Exception as exc:
        print("\n[main] Experiment orchestration failed:")
        traceback.print_exc()
        raise exc

    df = pd.DataFrame(results)
    print("\n=== Summary ===")
    print(df)

    summary_csv = results_dir / "results_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"[main] Saved summary CSV to {summary_csv}")

    plot_summary(df, results_dir)
    plot_convergence(convergence_logs, results_dir)


if __name__ == "__main__":
    main()

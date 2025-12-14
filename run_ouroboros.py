#!/usr/bin/env python3
"""Utility per eseguire inference Ouroboros3D su una singola immagine."""

import argparse
import datetime
import json
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional


def str2bool(value: str) -> bool:
    """Parse boolean values da stringa per argparse."""
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "1", "yes", "y"}:
        return True
    if lowered in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Valore booleano non valido: {value}")


def resolve_path(path_str: str, base_dir: Path) -> Path:
    """Restituisce path assoluto, considerando base_dir se path_str è relativo."""
    candidate = Path(path_str).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def run_and_log(cmd: List[str], cwd: Path, log_path: Path, verbose: bool = False) -> int:
    """Esegue il comando e streamma stdout/stderr su log_path."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    display_cmd = " ".join(shlex.quote(part) for part in cmd)
    if verbose:
        print(f"Eseguo comando: {display_cmd}")
    with log_path.open("w", encoding="utf-8") as log_file:
        timestamp = datetime.datetime.now(datetime.timezone.utc).isoformat()
        log_file.write(f"# Command: {display_cmd}\n")
        log_file.write(f"# Started: {timestamp}\n")
        log_file.flush()
        with subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as proc:
            if proc.stdout is None:
                raise RuntimeError("Impossibile catturare l'output del processo figlio")
            for line in proc.stdout:
                log_file.write(line)
                log_file.flush()
                if verbose:
                    print(line, end="")
            proc.wait()
            return proc.returncode


def find_latest_file(directory: Path, pattern: str) -> Optional[Path]:
    """Restituisce il file con mtime più recente che matcha il pattern."""
    latest_path: Optional[Path] = None
    latest_mtime = -1.0
    for path in directory.rglob(pattern):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def copy_overwrite(src: Path, dst: Path) -> None:
    """Copia src su dst sovrascrivendo l'eventuale file esistente."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def delete_glob(directory: Path, pattern: str) -> int:
    """Elimina i file che matchano pattern e restituisce quanti ne sono stati rimossi."""
    if not directory.exists():
        return 0
    removed = 0
    for path in directory.rglob(pattern):
        if not path.is_file():
            continue
        try:
            path.unlink()
            removed += 1
        except FileNotFoundError:
            continue
    return removed


def write_meta(meta_path: Path, payload: dict) -> None:
    """Scrive meta.json con indentazione standard."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def collect_torch_info() -> Optional[dict]:
    """Raccoglie info best-effort su torch/cuda senza far fallire lo script."""
    try:
        import torch
    except Exception:
        return None

    info = {"torch_version": getattr(torch, "__version__", "unknown")}
    try:
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = None
    info["cuda_available"] = cuda_available
    if cuda_available:
        try:
            device_count = torch.cuda.device_count()
            info["cuda_device_count"] = device_count
            info["cuda_devices"] = [torch.cuda.get_device_name(i) for i in range(device_count)]
        except Exception:
            pass
    return info


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run inference Ouroboros3D su singola immagine")
    parser.add_argument("--repo_root", required=True, help="Path alla repo Ouroboros3D")
    parser.add_argument("--input_image", required=True, help="Path immagine PNG/JPG da inferire")
    parser.add_argument("--case_dir", required=True, help="Directory dove salvare gli output del caso")
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoint/Ouroboros3D-SVD-LGM",
        help="Checkpoint da usare (relativo a repo_root se non assoluto)",
    )
    parser.add_argument(
        "--config_path",
        default="configs/mv/infer.yaml",
        help="Config da usare (relativo a repo_root se non assoluto)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed da passare all'inference")
    parser.add_argument(
        "--keep_png",
        type=str2bool,
        default=True,
        help="Se false elimina anche i PNG dal workspace",
    )
    parser.add_argument(
        "--verbose",
        type=str2bool,
        default=False,
        help="Logga su stdout l'output della pipeline",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    input_image = Path(args.input_image).expanduser().resolve()
    case_dir = Path(args.case_dir).expanduser().resolve()
    workspace_dir = case_dir / "workspace"
    case_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    log_path = case_dir / "infer.log"
    meta_path = case_dir / "meta.json"

    meta = {
        "repo_root": str(repo_root),
        "input_image": str(input_image),
        "case_dir": str(case_dir),
        "checkpoint_dir": None,
        "config_path": None,
        "seed": args.seed,
        "keep_png": bool(args.keep_png),
        "workspace_dir": str(workspace_dir),
        "infer_log": str(log_path),
        "command": [],
        "return_code": None,
        "inference_time_sec": None,
        "pred_ply": None,
        "pred_png": None,
        "removed_mp4_count": 0,
        "timestamp_utc": None,
        "error": None,
    }

    torch_info = collect_torch_info()
    if torch_info:
        meta["torch"] = torch_info

    try:
        if not repo_root.is_dir():
            raise FileNotFoundError(f"repo_root non esiste o non è una directory: {repo_root}")

        inference_py = repo_root / "inference.py"
        if not inference_py.is_file():
            raise FileNotFoundError(f"file inference.py non trovato in {repo_root}")

        if not input_image.is_file():
            raise FileNotFoundError(f"input_image non trovato: {input_image}")

        checkpoint_path = resolve_path(args.checkpoint_dir, repo_root)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"checkpoint non trovato: {checkpoint_path}")
        meta["checkpoint_dir"] = str(checkpoint_path)

        config_path = resolve_path(args.config_path, repo_root)
        if not config_path.exists():
            raise FileNotFoundError(f"config non trovato: {config_path}")
        meta["config_path"] = str(config_path)

        command = [
            sys.executable,
            "inference.py",
            "--input",
            str(input_image),
            "--checkpoint",
            str(checkpoint_path),
            "--output",
            str(workspace_dir),
            "--seed",
            str(args.seed),
            "--config",
            str(config_path),
        ]
        meta["command"] = command

        start = time.perf_counter()
        return_code = run_and_log(command, repo_root, log_path, verbose=bool(args.verbose))
        duration = time.perf_counter() - start
        meta["return_code"] = return_code
        meta["inference_time_sec"] = duration

        removed_mp4 = delete_glob(workspace_dir, "*.mp4")
        meta["removed_mp4_count"] = removed_mp4
        if args.verbose and removed_mp4:
            print(f"Rimossi {removed_mp4} file MP4 dal workspace")

        if return_code != 0:
            raise RuntimeError(
                f"Inference terminata con return code {return_code}. Consultare il log {log_path}"
            )

        latest_ply = find_latest_file(workspace_dir, "*.ply")
        if latest_ply is None:
            raise RuntimeError(
                f"Nessun file PLY generato in {workspace_dir}. Verificare il log per diagnosi."
            )

        pred_ply_path = case_dir / "pred.ply"
        copy_overwrite(latest_ply, pred_ply_path)
        meta["pred_ply"] = str(pred_ply_path)
        if args.verbose:
            print(f"Copiato PLY finale in {pred_ply_path}")

        if args.keep_png:
            latest_png = find_latest_file(workspace_dir, "*.png")
            if latest_png is not None:
                pred_png_path = case_dir / "pred.png"
                copy_overwrite(latest_png, pred_png_path)
                meta["pred_png"] = str(pred_png_path)
                if args.verbose:
                    print(f"Copiato PNG finale in {pred_png_path}")
            else:
                meta["pred_png"] = None
        else:
            removed_png = delete_glob(workspace_dir, "*.png")
            if args.verbose and removed_png:
                print(f"Rimossi {removed_png} file PNG dal workspace")
            meta["pred_png"] = None

    except Exception as exc:
        meta["error"] = str(exc)
        print(f"Errore: {exc}", file=sys.stderr)
        raise
    finally:
        meta["timestamp_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        write_meta(meta_path, meta)
        if args.verbose:
            print(f"Meta salvato in {meta_path}")

    if args.verbose:
        print(f"Inference completata in {meta['inference_time_sec']:.2f} secondi")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        sys.exit(1)

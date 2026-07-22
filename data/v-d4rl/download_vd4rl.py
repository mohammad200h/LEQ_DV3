#!/usr/bin/env python3
"""Download V-D4RL datasets for LEQ_DV3.

README: put V-D4RL data under data/v-d4rl/, then run hdf5_to_npz.py.

This script pulls the 84px HDF5 files (needed by hdf5_to_npz.py) into:
  data/v-d4rl/{task}/{diff}/84px/*.hdf5

Sources (prefer Hugging Face):
  - https://huggingface.co/datasets/conglu/vd4rl
  - https://github.com/conglu1997/v-d4rl  (Google Drive mirrors)

Examples:
  python download_vd4rl.py
  python download_vd4rl.py --tasks walker_walk --diffs medium_expert
  python download_vd4rl.py --source gdrive --tasks cheetah_run
  python download_vd4rl.py --convert
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Tasks / diffs used by hdf5_to_npz.py (and LEQ_DV3 training).
DEFAULT_TASKS = ("cheetah_run", "walker_walk")
DEFAULT_DIFFS = ("random", "medium", "medium_replay", "medium_expert")

HF_REPO = "conglu/vd4rl"

# Per-env zip archives (ReBRAC / community mirrors of the official Drive dump).
# https://github.com/DT6A/ReBRAC
GDRIVE_ZIPS = {
    "walker_walk": "1F4LIH_khOFw1asVvXo82OMa2tZ0Ax5Op",
    "cheetah_run": "1WR2LfK0y94C_1r2e1ps1dg6zSMHlVY_e",
    "humanoid_walk": "1zTBL8KWR3o07BQ62jJR7CeatN7vb-vjd",
}


def _ensure_pkg(import_name: str, pip_name: str | None = None) -> None:
  try:
    __import__(import_name)
  except ImportError:
    pkg = pip_name or import_name
    print(f"Installing {pkg} ...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def download_huggingface(tasks: list[str], diffs: list[str], out_dir: Path) -> None:
  _ensure_pkg("huggingface_hub")
  from huggingface_hub import snapshot_download

  patterns = [
      f"vd4rl/main/{task}/{diff}/84px/**"
      for task in tasks
      for diff in diffs
  ]
  print(f"Downloading from Hugging Face ({HF_REPO})")
  print(f"  patterns: {patterns}")
  cache = out_dir / "_hf_cache"
  snapshot_download(
      repo_id=HF_REPO,
      repo_type="dataset",
      allow_patterns=patterns,
      local_dir=str(cache),
  )
  src_root = cache / "vd4rl" / "main"
  if not src_root.exists():
    # Some HF layouts drop the outer vd4rl/ prefix.
    alt = cache / "main"
    src_root = alt if alt.exists() else src_root
  _promote_tree(src_root, out_dir, tasks, diffs, resolution="84px")


def download_gdrive(tasks: list[str], diffs: list[str], out_dir: Path) -> None:
  _ensure_pkg("gdown")
  import gdown

  raw = out_dir / "_gdrive_raw"
  raw.mkdir(parents=True, exist_ok=True)
  for task in tasks:
    file_id = GDRIVE_ZIPS.get(task)
    if file_id is None:
      raise ValueError(
          f"No Google Drive zip id for task={task!r}. "
          f"Known: {sorted(GDRIVE_ZIPS)}")
    zip_path = raw / f"{task}.zip"
    if not zip_path.exists():
      url = f"https://drive.google.com/uc?id={file_id}"
      print(f"Downloading {task} from Google Drive ({file_id})")
      gdown.download(url, str(zip_path), quiet=False)
    extract_dir = raw / task
    if not extract_dir.exists():
      print(f"Extracting {zip_path} ...")
      with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    # Zip layouts vary: try common roots.
    candidates = [
        extract_dir / "main" / task,
        extract_dir / task,
        extract_dir / "vd4rl" / "main" / task,
        extract_dir,
    ]
    src = next((p for p in candidates if _looks_like_task_dir(p, diffs)), None)
    if src is None:
      # Fall back: search for any */84px under extract_dir.
      hits = list(extract_dir.rglob("84px"))
      if not hits:
        raise FileNotFoundError(
            f"Could not find 84px data inside {extract_dir}")
      # Parent of 84px is {diff}; grandparent is task root.
      src = hits[0].parent.parent
    _promote_tree(
        src.parent,
        out_dir,
        [task],
        diffs,
        resolution="84px",
        task_src_override={task: src},
    )


def _looks_like_task_dir(path: Path, diffs: list[str]) -> bool:
  if not path.is_dir():
    return False
  return any((path / d / "84px").is_dir() for d in diffs)


def _promote_tree(
    src_root: Path,
    out_dir: Path,
    tasks: list[str],
    diffs: list[str],
    resolution: str,
    task_src_override: dict[str, Path] | None = None,
) -> None:
  """Copy/link {task}/{diff}/{resolution} into out_dir."""
  task_src_override = task_src_override or {}
  for task in tasks:
    task_src = task_src_override.get(task, src_root / task)
    for diff in diffs:
      src = task_src / diff / resolution
      dst = out_dir / task / diff / resolution
      if not src.exists():
        print(f"  skip missing: {src}")
        continue
      dst.parent.mkdir(parents=True, exist_ok=True)
      if dst.exists() or dst.is_symlink():
        if dst.is_dir() and not dst.is_symlink():
          shutil.rmtree(dst)
        else:
          dst.unlink()
      print(f"  {src} -> {dst}")
      try:
        dst.symlink_to(src.resolve(), target_is_directory=True)
      except OSError:
        shutil.copytree(src, dst)


def run_convert() -> None:
  script = ROOT / "hdf5_to_npz.py"
  print(f"Running conversion: {script}")
  subprocess.check_call([sys.executable, str(script)], cwd=str(ROOT))


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__,
                              formatter_class=argparse.RawDescriptionHelpFormatter)
  p.add_argument(
      "--source", choices=("hf", "gdrive"), default="hf",
      help="Download source (default: hf = Hugging Face).")
  p.add_argument(
      "--tasks", nargs="+", default=list(DEFAULT_TASKS),
      help=f"Tasks to download (default: {list(DEFAULT_TASKS)}).")
  p.add_argument(
      "--diffs", nargs="+", default=list(DEFAULT_DIFFS),
      help=f"Dataset splits (default: {list(DEFAULT_DIFFS)}).")
  p.add_argument(
      "--out-dir", type=Path, default=ROOT,
      help="Destination root (default: this directory).")
  p.add_argument(
      "--convert", action="store_true",
      help="After download, run hdf5_to_npz.py (84px -> 64px_dv3).")
  return p.parse_args()


def main() -> None:
  args = parse_args()
  out_dir = args.out_dir.resolve()
  out_dir.mkdir(parents=True, exist_ok=True)

  if args.source == "hf":
    download_huggingface(args.tasks, args.diffs, out_dir)
  else:
    download_gdrive(args.tasks, args.diffs, out_dir)

  print("Done. Expected layout:")
  print(f"  {out_dir}/<task>/<diff>/84px/*.hdf5")
  if args.convert:
    run_convert()
  else:
    print("Next: python hdf5_to_npz.py   # or re-run with --convert")


if __name__ == "__main__":
  main()

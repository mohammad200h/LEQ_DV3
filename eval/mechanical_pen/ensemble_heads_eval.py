#!/usr/bin/env python3
"""Measure how the world model's ensemble prior heads behave on held-out episodes.

For each world-model run, the model watches the first `report_openl_context` steps of a
window from a held-out episode (val_files.txt), then imagines the rest of the window with
the expert's actions, once per prior head held fixed (model_agent.visualize). From that:

  head_forces_rmse / head_positions_rmse   open-loop error of each head, in Xela counts / m
  worst_best_forces / worst_best_positions worst head's MSE over the best head's MSE
  spread_forces / spread_positions         how much the heads disagree: RMS over steps and
                                           taxels of the heads' std around their mean
  mean_forces_rmse / mean_positions_rmse   error of the heads' mean prediction
  spread_over_error_*                      spread / mean error. An ensemble whose heads
                                           disagree about as much as they are wrong gives
                                           about 1; much lower means it underestimates its
                                           own error.
  head_index_slope_*                       change in log MSE per head index. With the old
                                           chained heads, later heads are deeper, so a
                                           positive slope is expected there.

Writes <model_dir>/ensemble_heads.json.

Usage:
  python3 eval/mechanical_pen/ensemble_heads_eval.py --models <model_dir> [<model_dir> ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from visualize_model import (  # noqa: E402
    LeapModelVisualizer, _to_numpy, load_episode, slice_window)

WINDOW = 64


class HeadsProbe(LeapModelVisualizer):
  """The visualizer's model loading and jitted visualize, without the density cache."""

  def __init__(self, logdir: Path, data_dir: Path):
    self.logdir = logdir.expanduser().resolve()
    self.data_dir = data_dir.expanduser().resolve()
    self.window = WINDOW
    self.agent, self.config = self._load_agent()
    self.episodes = sorted(self.data_dir.glob('*.npz'))
    self._visualize = self._compile_visualize()

  def run(self, path: Path, start: int):
    episode = load_episode(path)
    batch, length = slice_window(episode, start, self.window)
    seed = self.agent._next_seeds(self.agent.train_sharded)
    carry = self.agent.init_report(1)
    import jax
    data = jax.device_put(self.agent._filter_data(batch), self.agent.train_sharded)
    outs, _ = self._visualize(self.agent.params, data, carry, seed)
    return _to_numpy(outs), length


def data_dir_for(model_dir: Path) -> Path:
  import yaml
  task = yaml.safe_load((model_dir / 'config.yaml').read_text())['task']
  return Path('data/leap') / task.removeprefix('leap_')


def evaluate(model_dir: Path, starts_per_episode: int) -> dict:
  probe = HeadsProbe(model_dir, data_dir_for(model_dir))
  num_obs = int(probe.config.report_openl_context)
  val = (model_dir / 'val_files.txt').read_text().split()
  head_mse = {'forces': [], 'positions': []}
  mean_mse = {'forces': [], 'positions': []}
  spread_var = {'forces': [], 'positions': []}
  for name in val:
    path = probe.data_dir / name
    total = len(np.load(path)['action'])
    if total < WINDOW:
      continue
    for start in np.linspace(0, total - WINDOW, starts_per_episode).astype(int):
      outs, length = probe.run(path, int(start))
      if 'forces_local_openl_heads' not in outs:
        raise SystemExit(f'{model_dir}: single prior, nothing to compare')
      steps = slice(num_obs, length)
      for key, field in (('forces', 'forces_local'), ('positions', 'positions')):
        heads = outs[f'{field}_openl_heads'][:, steps]  # (E, T, 368, 3)
        gt = outs[f'{field}_gt'][steps]
        head_mse[key].append(np.square(heads - gt[None]).mean(axis=(1, 2, 3)))
        mean = heads.mean(0)
        mean_mse[key].append(np.square(mean - gt).mean())
        spread_var[key].append(heads.var(0).mean())
  result = {'model_dir': str(model_dir), 'windows': len(mean_mse['forces']),
            'window': WINDOW, 'context': num_obs}
  for key in ('forces', 'positions'):
    per_head = np.mean(head_mse[key], axis=0)
    err = float(np.sqrt(np.mean(mean_mse[key])))
    spread = float(np.sqrt(np.mean(spread_var[key])))
    result[f'head_{key}_rmse'] = np.sqrt(per_head).tolist()
    result[f'worst_best_{key}'] = float(per_head.max() / per_head.min())
    result[f'mean_{key}_rmse'] = err
    result[f'spread_{key}'] = spread
    result[f'spread_over_error_{key}'] = spread / err
    result[f'head_index_slope_{key}'] = float(
        np.polyfit(np.arange(len(per_head)), np.log(per_head), 1)[0])
  return result


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--models', type=Path, nargs='+', required=True)
  parser.add_argument('--starts', type=int, default=4,
                      help='Windows per held-out episode, spread evenly')
  parser.add_argument('--redo', action='store_true')
  args = parser.parse_args()
  for model_dir in args.models:
    out = model_dir / 'ensemble_heads.json'
    if out.exists() and not args.redo:
      print(f'{out} exists, skipping')
      continue
    result = evaluate(model_dir, args.starts)
    out.write_text(json.dumps(result, indent=2))
    print(f'{model_dir}: spread/error forces {result["spread_over_error_forces"]:.3f}, '
          f'positions {result["spread_over_error_positions"]:.3f}, '
          f'worst/best forces {result["worst_best_forces"]:.2f}')


if __name__ == '__main__':
  main()

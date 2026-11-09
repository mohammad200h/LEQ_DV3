#!/usr/bin/env python3
"""Resample DreamerV3 demo chunks to a lower rate, in a new folder.

Examples (from LEQ_DV3/):
  python3 data/leap/resample_chunks.py                    # 20 Hz -> data/leap/mechanical_pen_20hz
  python3 data/leap/resample_chunks.py --hz 10            # 10 Hz -> data/leap/mechanical_pen_10hz

Then train on it with the folder's task name. Each rate needs its own world model:
  ./eval/mechanical_pen/train_model.sh 0 0 leap_mechanical_pen_20hz
  ./eval/leap/train_ditto.sh 0 0 leap_mechanical_pen_20hz

Each output chunk keeps the source chunk's time prefix, so files sort in the same order
and split_demos holds out the same episodes at every rate.

How each key is resampled (idx is the list of kept source steps):
- forces_local, positions: the value at each kept step, no averaging. A live policy also
  sees the latest reading.
- action: action'[i] is the command held when the next kept observation arrives,
  source action[idx[i + 1] - 1]. DreamerV3 reads action[t] as the action taken from obs[t],
  so this keeps each action leading to the following observation. The last one is copied.
- reward: summed over the dropped steps onto the next kept step, so returns are unchanged.
- is_first, is_last, is_terminal: set on the new first and last step.
- stepid: regenerated, with a new chunk uuid.

The first and last source steps are always kept, so the terminal reward survives. The last
gap can differ from the others by up to half a step.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pathlib
import string
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor

import numpy as np

BASE62 = string.digits + string.ascii_letters
BATCH_LENGTH = 65  # configs.yaml batch_length: shorter episodes give no training window
REQUIRED = ('action', 'reward', 'is_first', 'is_last', 'is_terminal')


def base62_uuid(value: bytes) -> str:
  """A 16 byte uuid as embodied's uuid.__str__ writes it (22 base62 characters)."""
  integer = int.from_bytes(value, 'big')
  chars = []
  while integer:
    chars.append(BASE62[integer % 62])
    integer //= 62
  return ''.join(reversed(chars)).rjust(22, '0')


def keep_indices(steps: int, ratio: float) -> np.ndarray:
  """Source steps to keep: every `ratio` steps from 0, always ending on the last step."""
  count = int(np.floor((steps - 1) / ratio)) + 1
  idx = np.round(np.arange(count) * ratio).astype(np.int64)
  idx = idx[idx < steps - 1]
  if len(idx) > 1 and steps - 1 - idx[-1] < ratio / 2:
    idx = idx[:-1]  # Replace the last kept step instead of adding a very short final gap.
  return np.concatenate([idx, [steps - 1]])


def resample_episode(episode: dict[str, np.ndarray], ratio: float) -> dict[str, np.ndarray]:
  """The episode at 1/ratio of its rate, without stepid."""
  missing = [key for key in REQUIRED if key not in episode]
  if missing:
    raise ValueError(f'missing keys: {missing}')
  steps = len(episode['action'])
  idx = keep_indices(steps, ratio)
  out = {
      key: value[idx] for key, value in episode.items()
      if key not in REQUIRED and key != 'stepid' and len(value) == steps}
  action = episode['action'][idx]
  action[:-1] = episode['action'][idx[1:] - 1]
  out['action'] = action
  # Segments: [0], (idx[0], idx[1]], (idx[1], idx[2]], ...
  starts = np.concatenate([[0], idx[:-1] + 1])
  out['reward'] = np.add.reduceat(episode['reward'], starts).astype(episode['reward'].dtype)
  for key in ('is_first', 'is_last', 'is_terminal'):
    out[key] = np.zeros(len(idx), bool)
  out['is_first'][0] = True
  out['is_last'][-1] = True
  out['is_terminal'][-1] = bool(episode['is_terminal'][-1])
  return out


def force_moments(forces: np.ndarray) -> np.ndarray:
  """Per-axis count, sum and squared sum of non-zero values (zeros are dead taxels)."""
  moments = np.zeros((3, 3), np.float64)
  for axis in range(3):
    values = forces[..., axis]
    values = values[values != 0].astype(np.float64)
    moments[axis] = len(values), values.sum(), np.square(values).sum()
  return moments


def convert_file(job: tuple[str, str, float]) -> dict:
  source, out_dir, ratio = job
  with np.load(source) as raw:
    episode = {key: raw[key] for key in raw.files}
  chunk = resample_episode(episode, ratio)
  steps = len(chunk['action'])
  chunk_id = uuid.uuid4().bytes
  chunk['stepid'] = np.array(
      [np.frombuffer(chunk_id + i.to_bytes(4, 'big'), np.uint8) for i in range(steps)])
  stamp = os.path.basename(source).split('-', 1)[0]
  name = f'{stamp}-{base62_uuid(chunk_id)}-{base62_uuid(bytes(16))}-{steps}.npz'
  np.savez_compressed(os.path.join(out_dir, name), **chunk)
  return {
      'source': os.path.basename(source),
      'name': name,
      'steps': steps,
      'source_steps': len(episode['action']),
      'reward_sum': float(chunk['reward'].sum()),
      'source_reward_sum': float(episode['reward'].sum()),
      'moments': force_moments(chunk['forces_local']) if 'forces_local' in chunk else None,
  }


def source_rate(input_dir: pathlib.Path, override: float | None) -> float:
  if override is not None:
    return override
  try:
    return float(json.loads((input_dir / 'manifest.json').read_text())['freq'])
  except (OSError, ValueError, KeyError, TypeError):
    raise SystemExit(
        f'No freq in {input_dir}/manifest.json. Give the rate with --source-hz.') from None


def rate_label(hz: float) -> str:
  return f'{hz:g}'


def prepare_output(out_dir: pathlib.Path, input_dir: pathlib.Path, overwrite: bool) -> None:
  if out_dir.resolve() == input_dir.resolve():
    raise SystemExit('--output is the input folder. The source chunks must stay untouched.')
  existing = list(out_dir.glob('*.npz')) if out_dir.exists() else []
  if existing and not overwrite:
    raise SystemExit(
        f'{out_dir} already has {len(existing)} chunks. Pass --overwrite to replace them.')
  out_dir.mkdir(parents=True, exist_ok=True)
  for path in existing:
    path.unlink()


def main() -> None:
  here = pathlib.Path(__file__).resolve().parent
  parser = argparse.ArgumentParser(
      description='Resample DreamerV3 demo chunks to a lower rate, in a new folder.')
  parser.add_argument('--hz', type=float, default=20.0, help='Target rate (default: 20)')
  parser.add_argument(
      '--input', type=pathlib.Path, default=here / 'mechanical_pen',
      help='Folder of source chunks (default: data/leap/mechanical_pen)')
  parser.add_argument(
      '--output', type=pathlib.Path, default=None,
      help='Output folder (default: <input>_<hz>hz, next to the input)')
  parser.add_argument(
      '--source-hz', type=float, default=None,
      help='Source rate (default: freq in the input manifest.json)')
  parser.add_argument('--overwrite', action='store_true', help='Replace chunks in --output')
  parser.add_argument('--workers', type=int, default=min(8, os.cpu_count() or 1))
  args = parser.parse_args()

  sources = sorted(args.input.glob('*.npz'))
  if not sources:
    raise SystemExit(f'No .npz chunks in {args.input}')
  rate = source_rate(args.input, args.source_hz)
  if not 0 < args.hz < rate:
    raise SystemExit(f'--hz must be above 0 and below the source rate ({rate:g} Hz).')
  # Don't resolve the input: data/leap/mechanical_pen is a symlink, and the output and the
  # task name should follow the name used here.
  out_dir = args.output or args.input.parent / f'{args.input.name}_{rate_label(args.hz)}hz'
  prepare_output(out_dir, args.input, args.overwrite)

  ratio = rate / args.hz
  print(f'{len(sources)} chunks, {rate:g} Hz -> {args.hz:g} Hz (keep 1 in {ratio:g}) -> {out_dir}')
  jobs = [(str(path), str(out_dir), ratio) for path in sources]
  results = []
  with ProcessPoolExecutor(args.workers) as pool:
    for done, result in enumerate(pool.map(convert_file, jobs), start=1):
      results.append(result)
      if done % 50 == 0 or done == len(jobs):
        print(f'  {done}/{len(jobs)}', flush=True)

  steps = [r['steps'] for r in results]
  source_steps = sum(r['source_steps'] for r in results)
  short = [r['name'] for r in results if r['steps'] < BATCH_LENGTH]
  changed = [r['name'] for r in results if abs(r['reward_sum'] - r['source_reward_sum']) > 1e-3]
  if changed:
    raise SystemExit(f'Episode return changed in {len(changed)} chunks, e.g. {changed[0]}')

  manifest = {
      'created': datetime.datetime.now().isoformat(timespec='seconds'),
      'note': 'DreamerV3 chunks, already converted: do not run convert_episodes.py on them',
      'source': str(args.input.resolve()),
      'source_freq': rate,
      'freq': args.hz,
      'written': {r['source']: r['name'] for r in results},
      'steps': {'min': min(steps), 'max': max(steps), 'total': sum(steps)},
      f'shorter_than_{BATCH_LENGTH}': short,
  }
  moments = [r['moments'] for r in results if r['moments'] is not None]
  if moments:
    count, total, squares = np.sum(moments, axis=0).T
    mean = total / count
    std = np.sqrt(squares / count - np.square(mean))
    manifest['forces_nonzero'] = {'mean': mean.round(4).tolist(), 'std': std.round(4).tolist()}
  (out_dir / 'manifest.json').write_text(json.dumps(manifest, indent=1))

  print(f'Steps: {source_steps} -> {sum(steps)} (per episode {min(steps)} to {max(steps)})')
  if 'forces_nonzero' in manifest:
    print(f'Force stats, non-zero values: {manifest["forces_nonzero"]}')
    print('  Compare with forces_norm in dreamerv3/configs.yaml before training.')
  if short:
    print(
        f'Warning: {len(short)} episodes are shorter than batch_length {BATCH_LENGTH} '
        'and give no training window.', file=sys.stderr)
  task = out_dir.name
  print(f'Train with task leap_{task}, e.g. ./eval/mechanical_pen/train_model.sh 0 0 leap_{task}')


if __name__ == '__main__':
  main()

"""Convert sparsh-skin-sim collection ``.npz`` episodes to DreamerV3 replay chunks.

Collection episodes (from ``data_collection.py``) store ARS pairs
``(action_t, obs_{t+1}, reward_{t+1})``. DreamerV3 expects RSA, so actions
are shifted forward by one step (same as ``data/v-d4rl/hdf5_to_npz.py``).
Reward stays aligned with the observation it was received with.

Observations written per step:
  - forces_local        (368, 3)  taxel forces in the FK/taxel frame
  - positions           (368, 3)  taxel positions in the FK frame
  - action              (16,)     joint targets, scaled to [-1, 1]
  - reward, is_first, is_last, is_terminal, stepid

Example:
  python data/leap/convert_episodes.py \\
      --input /workspace/sparsh-skin-sim/data/finger_close \\
      --output data/leap/finger_close
"""

from __future__ import annotations

import argparse
import datetime
import io
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import embodied  # noqa: E402
from embodied.envs.leap import (  # noqa: E402
    ACTION_HIGH,
    ACTION_LOW,
    N_JOINTS,
    N_TAXELS,
    normalize_action,
)


def save_episode(directory, episode, num, chunk_uuid):
  timestamp = (
      datetime.datetime.now() + datetime.timedelta(0, num)
  ).strftime('%Y%m%dT%H%M%S')
  succ_uuid = embodied.uuid(0)
  length = len(episode['action'])
  filename = directory / f'{timestamp}-{chunk_uuid}-{succ_uuid}-{length}.npz'

  step_id = np.array([
      np.frombuffer(chunk_uuid.value + i.to_bytes(4, 'big'), np.uint8)
      for i in range(length)
  ])
  episode['stepid'] = step_id
  with io.BytesIO() as stream:
    np.savez_compressed(stream, **episode)
    stream.seek(0)
    filename.write_bytes(stream.read())
  return filename


def _load_metadata(raw):
  if 'metadata_json' not in raw.files:
    return {}
  payload = raw['metadata_json']
  if payload.shape == ():
    payload = payload.item()
  if isinstance(payload, bytes):
    payload = payload.decode('utf-8')
  return json.loads(str(payload))


def convert_episode(path):
  raw = np.load(path, allow_pickle=True)
  metadata = _load_metadata(raw)

  actions = np.asarray(raw['actions'], dtype=np.float32)
  if actions.ndim != 2 or actions.shape[-1] != N_JOINTS:
    raise ValueError(
        f'{path}: expected actions (T, {N_JOINTS}), got {actions.shape}')

  forces = np.asarray(raw['flex_taxel_fk/forces_local'], dtype=np.float32)
  positions = np.asarray(raw['flex_taxel_fk/positions'], dtype=np.float32)
  if forces.shape[1:] != (N_TAXELS, 3):
    raise ValueError(
        f'{path}: expected forces_local (T, {N_TAXELS}, 3), got {forces.shape}')
  if positions.shape[1:] != (N_TAXELS, 3):
    raise ValueError(
        f'{path}: expected positions (T, {N_TAXELS}, 3), got {positions.shape}')

  n = int(actions.shape[0])
  if forces.shape[0] != n or positions.shape[0] != n:
    raise ValueError(
        f'{path}: length mismatch actions={n} forces={forces.shape[0]} '
        f'positions={positions.shape[0]}')

  if 'rewards' in raw.files:
    reward = np.asarray(raw['rewards'], dtype=np.float32).reshape(-1)
    if reward.shape[0] != n:
      raise ValueError(
          f'{path}: reward length {reward.shape[0]} != actions {n}')
  else:
    print(f'{path}: no rewards array; writing zeros', file=sys.stderr)
    reward = np.zeros(n, dtype=np.float32)

  # ARS (collection) -> RSA (DreamerV3): action[t] is taken from obs[t].
  # Reward stays with the observation it was received with (no shift).
  actions = np.concatenate([actions[1:], np.zeros_like(actions[:1])], axis=0)
  actions = normalize_action(actions, ACTION_LOW, ACTION_HIGH)

  is_first = np.zeros(n, dtype=bool)
  is_last = np.zeros(n, dtype=bool)
  is_terminal = np.zeros(n, dtype=bool)
  is_first[0] = True
  is_last[-1] = True
  is_terminal[-1] = bool(metadata.get('terminated_by_env', False))

  return {
      'forces_local': forces,
      'positions': positions,
      'action': actions,
      'reward': reward,
      'is_first': is_first,
      'is_last': is_last,
      'is_terminal': is_terminal,
  }, n, metadata


def main():
  parser = argparse.ArgumentParser(
      description='Convert collection npz episodes to DreamerV3 replay chunks.')
  parser.add_argument(
      '--input', type=Path, required=True,
      help='Directory of episode_XXXX.npz files from data_collection.py')
  parser.add_argument(
      '--output', type=Path, default=None,
      help='Output directory (default: LEQ_DV3/data/leap/<input.name>)')
  args = parser.parse_args()

  input_dir = args.input.expanduser().resolve()
  if not input_dir.is_dir():
    raise FileNotFoundError(f'Input directory not found: {input_dir}')

  output_dir = args.output
  if output_dir is None:
    output_dir = _REPO_ROOT / 'data' / 'leap' / input_dir.name
  output_dir = output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  episodes = sorted(input_dir.glob('episode_*.npz'))
  if not episodes:
    raise FileNotFoundError(f'No episode_*.npz files in {input_dir}')

  lengths = []
  reward_mins = []
  reward_maxs = []
  for i, path in enumerate(episodes):
    episode, length, metadata = convert_episode(path)
    chunk_uuid = embodied.uuid()
    out = save_episode(output_dir, episode, i, chunk_uuid)
    lengths.append(length)
    rew = episode['reward']
    reward_mins.append(float(rew.min()))
    reward_maxs.append(float(rew.max()))
    print(
        f'{path.name} -> {out.name}  steps={length}  '
        f'reward=[{rew.min():.4g}, {rew.max():.4g}]  '
        f'terminated_by_env={metadata.get("terminated_by_env")}')

  print(
      f'Wrote {len(episodes)} chunks to {output_dir}  '
      f'(steps min/mean/max={min(lengths)}/{sum(lengths)/len(lengths):.1f}/'
      f'{max(lengths)}; reward min/max='
      f'{min(reward_mins):.4g}/{max(reward_maxs):.4g})')
  if max(reward_maxs) == 0.0 and min(reward_mins) == 0.0:
    print(
        'Warning: all converted rewards are 0. Re-collect with the updated '
        'data_collection.py so episodes include a rewards array.',
        file=sys.stderr)
  short = [n for n in lengths if n < 65]
  if short:
    print(
        f'Warning: {len(short)} episodes are shorter than DreamerV3 '
        f'batch_length=65 and will not yield training windows.')


if __name__ == '__main__':
  main()

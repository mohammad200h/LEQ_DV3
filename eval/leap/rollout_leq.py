#!/usr/bin/env python3
"""Roll out a trained LEQ policy in sparsh-skin-sim LeapFlexEnv / LeapFlexCyringeEnv.

Example:
  python eval/leap/rollout_leq.py \\
      --logdir logdir/leq/leap_squeeze/0.1/1 \\
      --env-config /workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml

  python eval/leap/rollout_leq.py \\
      --logdir logdir/leq/leap_cyringe/0.1/0 \\
      --env-config /workspace/sparsh-skin-sim/data_collection/config/cyringe.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import embodied  # noqa: E402
from dreamerv3.main import make_agent  # noqa: E402
from embodied.envs.leap import ACTION_HIGH, ACTION_LOW  # noqa: E402


def _setup_sparsh(sparsh_root: Path) -> None:
  sparsh_root = sparsh_root.expanduser().resolve()
  if not sparsh_root.is_dir():
    raise FileNotFoundError(f'sparsh-skin-sim root not found: {sparsh_root}')
  root = str(sparsh_root)
  if root not in sys.path:
    sys.path.insert(0, root)


def load_agent(logdir: Path):
  logdir = logdir.expanduser().resolve()
  config_path = logdir / 'config.yaml'
  if not config_path.exists():
    raise FileNotFoundError(f'Missing config: {config_path}')
  ckpt_path = logdir / 'checkpoint.ckpt'
  if not ckpt_path.exists():
    raise FileNotFoundError(f'Missing checkpoint: {ckpt_path}')

  from ruamel.yaml import YAML
  yaml = YAML(typ='safe')
  config = embodied.Config(yaml.load(config_path.read_text()))
  config = config.update(logdir=str(logdir))
  agent = make_agent(config)

  checkpoint = embodied.Checkpoint(ckpt_path)
  checkpoint.agent = agent
  checkpoint.load(keys=['agent'])
  return agent, config


def _is_cyringe_config(env_cfg: dict, policy_cfg: dict) -> bool:
  if 'cyringe' in env_cfg or 'handle_site' in policy_cfg:
    return True
  return 'grasp_pattern' not in policy_cfg


def _load_env_config(env_config: Path) -> dict:
  import yaml
  from env.domain_randomization import DomainRandomizationConfig
  from util.trajectory_generation import DEFAULT_TRAJ_WAYPOINTS

  with env_config.open(encoding='utf-8') as handle:
    config = yaml.safe_load(handle)

  collection = config['collection']
  env_cfg = config['env']
  policy_cfg = config.get('policy') or {}

  n_substeps = int(env_cfg['n_substeps'])
  num_envs = int(env_cfg.get('num_envs', 1))
  kind = 'cyringe' if _is_cyringe_config(env_cfg, policy_cfg) else 'squeeze'

  if kind == 'cyringe':
    max_timestep = env_cfg.get('max_timestep')
    if max_timestep is not None:
      max_timestep = int(max_timestep)
    n_traj_waypoints = None
    if bool(policy_cfg.get('use_trajectory', False)):
      n_traj_waypoints = int(policy_cfg.get('traj_waypoints') or DEFAULT_TRAJ_WAYPOINTS)
    env_spec = {
        'n_substeps': n_substeps,
        'max_timestep': max_timestep,
        'cyringe': dict(env_cfg.get('cyringe') or {}) or None,
        'ik_helpers': True,
        'n_traj_waypoints': n_traj_waypoints,
    }
    return {
        'kind': kind,
        'duration': float(collection['duration']),
        'max_timestep': max_timestep,
        'num_envs': num_envs,
        'env_spec': env_spec,
    }

  joint_movement_threshold = env_cfg.get('joint_movement_threshold')
  if joint_movement_threshold is not None:
    joint_movement_threshold = float(joint_movement_threshold)

  domain_randomization = DomainRandomizationConfig.from_dict(
      env_cfg.get('domain_randomization'))
  tetris_offset = env_cfg.get('tetris_offset', [0.0, 0.0, 0.0])

  env_spec = {
      'n_substeps': n_substeps,
      'spawn_tetris': bool(env_cfg.get('spawn_tetris', False)),
      'tetris_shape': str(env_cfg.get('tetris_shape', 'T')),
      'tetris_scale': float(env_cfg.get('tetris_scale', 1.5)),
      'tetris_offset': tuple(float(v) for v in tetris_offset),
      'tetris_flex': env_cfg.get('tetris_flex'),
      'joint_movement_threshold': joint_movement_threshold,
      'domain_randomization': domain_randomization,
      'motion_type': str(policy_cfg['grasp_pattern']),
  }
  return {
      'kind': kind,
      'duration': float(collection['duration']),
      'max_timestep': None,
      'num_envs': num_envs,
      'env_spec': env_spec,
      'grasp_pattern': str(policy_cfg['grasp_pattern']),
  }


def build_env(env_config: Path, sparsh_root: Path):
  _setup_sparsh(sparsh_root)
  from env.leap_flex_cyringe_env import LeapFlexCyringeEnv
  from env.leap_flex_env import LeapFlexEnv

  cfg = _load_env_config(env_config)
  if cfg['kind'] == 'cyringe':
    env = LeapFlexCyringeEnv(**cfg['env_spec'])
  else:
    env = LeapFlexEnv(**cfg['env_spec'])
  duration = float(cfg['duration'])
  step_seconds = float(env.model.opt.timestep * env._n_substeps)
  max_steps = max(1, int(np.ceil(duration / step_seconds)))
  max_timestep = cfg.get('max_timestep')
  if max_timestep is not None:
    max_steps = min(max_steps, int(max_timestep))
  return env, max_steps, cfg


def denormalize_action(action, ctrl_lo, ctrl_hi):
  action = np.asarray(action, dtype=np.float32).reshape(-1)
  scaled = (action + 1.0) / 2.0 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
  return np.clip(scaled, ctrl_lo, ctrl_hi)


def to_agent_obs(
    raw,
    *,
    reward=0.0,
    is_first=False,
    is_last=False,
    is_terminal=False,
):
  fk = raw['flex_taxel_fk']
  return {
      'forces_local': np.asarray(fk['forces_local'], dtype=np.float32),
      'positions': np.asarray(fk['positions'], dtype=np.float32),
      'reward': np.float32(reward),
      'is_first': bool(is_first),
      'is_last': bool(is_last),
      'is_terminal': bool(is_terminal),
  }


def batch_obs(obs):
  batched = {}
  for key, value in obs.items():
    value = np.asarray(value)
    if value.ndim == 0:
      dtype = np.float32 if key == 'reward' else value.dtype
      batched[key] = np.array([value], dtype=dtype)
    elif key in ('is_first', 'is_last', 'is_terminal'):
      batched[key] = np.array([bool(value)], dtype=bool)
    else:
      batched[key] = value[np.newaxis, ...].astype(np.float32, copy=False)
  return batched


def rollout_episode(
    agent,
    env,
    max_steps: int,
    *,
    seed: int | None = None,
    viewer=None,
):
  carry = agent.init_policy(1)
  raw, _info = env.reset(seed=seed)
  obs = to_agent_obs(raw, is_first=True)
  total_reward = 0.0
  step_rewards = []
  step_seconds = float(env.model.opt.timestep * env._n_substeps)
  steps_done = 0

  for step in range(max_steps):
    if viewer is not None and not viewer.is_running():
      break

    step_start = time.time()
    acts, _outs, carry = agent.policy(batch_obs(obs), carry, mode='eval')
    action = np.asarray(acts['action'][0], dtype=np.float32)
    action = denormalize_action(action, env._ctrl_lo, env._ctrl_hi)

    raw, reward, terminated, truncated, _info = env.step(action)
    total_reward += float(reward)
    step_rewards.append(float(reward))
    steps_done = step + 1
    done = bool(terminated or truncated)
    obs = to_agent_obs(
        raw,
        reward=reward,
        is_last=done,
        is_terminal=terminated,
    )
    print(
        f'  step {step:4d}: reward={reward:.3f} total={total_reward:.3f} '
        f'done={done}')

    if viewer is not None:
      viewer.sync()
      leftover = step_seconds - (time.time() - step_start)
      if leftover > 0:
        time.sleep(leftover)

    if done:
      break

  return {
      'steps': steps_done,
      'total_reward': total_reward,
      'mean_reward': float(np.mean(step_rewards)) if step_rewards else 0.0,
      'max_reward': float(np.max(step_rewards)) if step_rewards else 0.0,
  }


def main():
  parser = argparse.ArgumentParser(
      description='Roll out a trained LEQ policy in LeapFlexEnv.')
  parser.add_argument(
      '--logdir', type=Path, required=True,
      help='LEQ logdir containing config.yaml and checkpoint.ckpt')
  parser.add_argument(
      '--env-config', type=Path,
      default=Path('/workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml'),
      help='sparsh-skin-sim collection YAML used during demo collection')
  parser.add_argument(
      '--sparsh-root', type=Path, default=Path('/workspace/sparsh-skin-sim'),
      help='Path to sparsh-skin-sim checkout')
  parser.add_argument('--episodes', type=int, default=1)
  parser.add_argument(
      '--max-steps', type=int, default=None,
      help='Override max steps per episode (default: from env config duration)')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument(
      '--render', action='store_true',
      help='Open a MuJoCo passive viewer (single env only)')
  args = parser.parse_args()

  agent, config = load_agent(args.logdir)
  env, default_max_steps, env_cfg = build_env(args.env_config, args.sparsh_root)
  max_steps = args.max_steps or default_max_steps

  print(f'Logdir:     {args.logdir.resolve()}')
  print(f'Env config: {args.env_config.resolve()}')
  print(f'Env kind:   {env_cfg["kind"]}')
  print(f'Task:       {config.task}')
  print(f'Algo:       {config.algo}')
  print(f'Max steps:  {max_steps}')
  print(f'Episodes:   {args.episodes}')
  print(f'Render:     {args.render}')

  if args.render:
    import mujoco.viewer
    viewer_ctx = mujoco.viewer.launch_passive(env.model, env.data)
  else:
    viewer_ctx = nullcontext()

  results = []
  try:
    with viewer_ctx as viewer:
      for episode in range(args.episodes):
        if viewer is not None and not viewer.is_running():
          print('Viewer closed; stopping early.')
          break
        episode_seed = None if args.seed is None else args.seed + episode
        print(f'\nEpisode {episode + 1}/{args.episodes}')
        results.append(rollout_episode(
            agent, env, max_steps, seed=episode_seed, viewer=viewer))
  finally:
    env.close()

  totals = [r['total_reward'] for r in results]
  print('\nSummary')
  print(f'  episodes: {len(results)}')
  if not results:
    print('  no completed episodes')
    return
  print(f'  total reward mean/std: {np.mean(totals):.3f} / {np.std(totals):.3f}')
  print(f'  total reward min/max:  {np.min(totals):.3f} / {np.max(totals):.3f}')
  print(f'  mean step reward:      {np.mean([r["mean_reward"] for r in results]):.3f}')


if __name__ == '__main__':
  main()

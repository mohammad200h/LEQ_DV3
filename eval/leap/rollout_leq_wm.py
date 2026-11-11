#!/usr/bin/env python3
"""Run a trained policy inside its world model and decode what it imagines.

No simulator or hand is involved. For one demo episode:
  1. The world model watches the real episode up to --start.
  2. From that state it imagines the rest twice (prior): once with the policy choosing every
     action (mode, as at deployment), and once replaying the expert's actions.
  3. Separately, the posterior keeps filtering the real episode: each step it gets the real
     observation and the expert's previous action. The policy is queried on every posterior
     state too, without its action being used, which shows whether the policy is jittery even
     on real states.
  4. All of these are decoded to taxel forces and positions and compared with the real episode.

Replaying the expert's actions shows the world model's own error. Only the gap between
the policy and that reference comes from the policy. This shows what the world model
predicts, not what would happen on the hand.

Writes to <logdir>/rollout_wm/<episode>/ unless --out is set:
  rollout.gif    real, posterior, prior with expert actions and prior with the policy, side by
                 side and coloured by ||force||, with command and taxel jitter traces below
  summary.png    joint commands and error curves over time
  rollout.npz    every array behind the plots
  metrics.json   the numbers printed at the end

Example:
  python3 eval/leap/rollout_leq_wm.py --logdir logdir/ditto/leap_mechanical_pen/0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
# The container's Debian mpl_toolkits shadows the copy that matches pip's matplotlib, and
# importing the Debian one fails. Look in matplotlib's own install first.
import mpl_toolkits  # noqa: E402
mpl_toolkits.__path__.insert(0, str(Path(matplotlib.__file__).parents[1] / 'mpl_toolkits'))
import mpl_toolkits.mplot3d  # noqa: E402,F401
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

from dreamerv3 import jaxutils  # noqa: E402
from dreamerv3 import ninjax as nj  # noqa: E402
from embodied.envs.leap import ACTION_HIGH, ACTION_LOW  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rollout_leq import load_agent  # noqa: E402

OBS_KEYS = ('forces_local', 'positions', 'reward', 'is_first', 'is_last', 'is_terminal')
STREAMS = {
    'real': ('Real episode', 'black'),
    'posterior': ('Posterior: real obs + expert actions', 'tab:green'),
    'expert': ('Prior: expert actions', 'tab:blue'),
    'policy': ('Prior: policy actions', 'tab:red'),
}
DECODED = ('posterior', 'expert', 'policy')


def to_numpy(tree):
  # The agent turns on JAX's transfer guard, so reads from the device must be explicit.
  with jax._src.config.explicit_device_get_scope():
    return jax.tree_util.tree_map(np.asarray, jax.device_get(tree))


class WorldModelRollout:
  """One-step observe and imagine functions on the agent's loaded parameters."""

  def __init__(self, agent, config):
    self.agent = agent
    self.inner = agent.agent
    self.norm = config.forces_norm
    if not hasattr(self.inner, 'actor'):
      raise ValueError(f'algo "{config.algo}" has no policy to roll out.')
    self._observe = self._compile(self._observe_step)
    self._imagine = self._compile(self._imagine_step)

  def _compile(self, fn):
    pure = nj.pure(fn)

    def wrapped(params, *args, seed):
      _, outs = pure(params, *args, seed=seed, create=True)
      return outs

    return jax.jit(wrapped)

  def _policy_action(self, out):
    inner = self.inner
    dist = inner.actor(out, bdims=1)['action']
    action = dist.mode()
    if inner.actor_dists['action'] == 'tanh_normal':
      action = jnp.tanh(action)
    return action

  def _outputs(self, lat, out):
    inner = self.inner
    dec = inner.dec(out, bdims=1)
    return lat, {
        'policy_action': self._policy_action(out).astype(jnp.float32),
        'forces': jaxutils.denorm_forces(dec['forces_local'].mode(), self.norm).astype(jnp.float32),
        'positions': dec['positions'].mode().astype(jnp.float32),
        'reward': inner.rew(out, bdims=1).mode().astype(jnp.float32),
        'cont': inner.con(out, bdims=1).mean().astype(jnp.float32),
        'deter': out['deter'].astype(jnp.float32),
    }

  def _observe_step(self, lat, prevact, obs):
    inner = self.inner
    obs = inner.preprocess(obs)
    embed = inner.enc(obs, bdims=1)
    prevact = jaxutils.onehot_dict({'action': prevact}, inner.act_space)
    lat, out = inner.dyn.observe(lat, prevact, embed, obs['is_first'], bdims=1)
    return self._outputs(lat, out)

  def _imagine_step(self, lat, action):
    lat, out = self.inner.dyn.imagine(lat, {'action': action}, bdims=1)
    return self._outputs(lat, out)

  def _seed(self):
    return self.agent._next_seeds(self.agent.train_sharded)

  def observe(self, lat, prevact, obs):
    return self._observe(
        self.agent.params, lat, jax.device_put(prevact), jax.device_put(obs), seed=self._seed())

  def imagine(self, lat, action):
    return self._imagine(self.agent.params, lat, jax.device_put(action), seed=self._seed())

  def initial_latent(self):
    return self.agent.init_report(1)[0]


def run_episode(model, episode, start, end):
  """Decoded real, policy and expert-action streams for steps start to end - 1."""
  expert_actions = episode['action'].astype(np.float32)
  lat = model.initial_latent()
  prevact = np.zeros((1, expert_actions.shape[-1]), np.float32)
  posterior = []
  for t in range(end):
    obs = {key: np.asarray(episode[key][t])[None] for key in OBS_KEYS}
    lat, outs = model.observe(lat, prevact, obs)
    posterior.append(to_numpy(outs))
    prevact = expert_actions[t][None]
    if t == start:
      start_lat = lat
  posterior = stack(posterior)
  streams = {'posterior': posterior}

  # Both imagined streams start from the same real state, so their index 0 is the posterior.
  first = {key: value[start] for key, value in posterior.items()}
  for name in ('policy', 'expert'):
    lat, outs = start_lat, first
    steps = [{key: value[None] for key, value in first.items()}]
    actions = []
    for t in range(start, end - 1):
      action = outs['policy_action'] if name == 'policy' else expert_actions[t]
      action = np.asarray(action, np.float32).reshape(1, -1)
      actions.append(action[0])
      lat, step = model.imagine(lat, action)
      outs = {key: value[0] for key, value in to_numpy(step).items()}
      steps.append({key: value[None] for key, value in outs.items()})
    final = outs['policy_action'] if name == 'policy' else expert_actions[end - 1]
    actions.append(np.asarray(final, np.float32).reshape(-1))
    stream = stack(steps)
    stream['action'] = np.stack(actions)
    streams[name] = stream
  return streams


def stack(steps):
  return {key: np.concatenate([step[key] for step in steps]) for key in steps[0]}


def latent_reward(learner, expert):
  """DITTO's reward (paper eq. 8) on the deterministic state."""
  dot = np.sum(learner * expert, -1)
  denom = np.maximum(np.linalg.norm(learner, axis=-1), np.linalg.norm(expert, axis=-1)) ** 2
  return dot / np.maximum(denom, 1e-8)


def to_radians(action):
  return (action + 1.0) / 2.0 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW


def compute_curves(episode, streams, start, end):
  real_forces = episode['forces_local'][start:end].astype(np.float32)
  real_positions = episode['positions'][start:end].astype(np.float32)
  expert_actions = episode['action'][start:end].astype(np.float32)
  expert_deter = streams['posterior']['deter'][start:end]
  # Mean taxel movement between consecutive frames, in mm. Jitter in the decoded hand shows
  # up here as movement the real hand doesn't make.
  position_step = lambda p: 1000.0 * np.linalg.norm(np.diff(p, axis=0), axis=-1).mean(-1)
  curves = {
      'real_force_norm': np.linalg.norm(real_forces, axis=-1).mean(-1),
      'real_position_step_mm': position_step(real_positions),
      'expert_action_step': np.abs(np.diff(expert_actions[:-1], axis=0)).mean(-1),
  }
  posterior = {key: value[start:end] for key, value in streams['posterior'].items()}
  for name in DECODED:
    stream = posterior if name == 'posterior' else streams[name]
    curves[f'{name}_position_step_mm'] = position_step(stream['positions'])
    curves[f'{name}_force_norm'] = np.linalg.norm(stream['forces'], axis=-1).mean(-1)
    curves[f'{name}_force_rmse'] = np.sqrt(np.square(stream['forces'] - real_forces).mean((1, 2)))
    curves[f'{name}_position_error_mm'] = 1000.0 * np.linalg.norm(
        stream['positions'] - real_positions, axis=-1).mean(-1)
    curves[f'{name}_latent_reward'] = latent_reward(stream['deter'], expert_deter)
    curves[f'{name}_reward'] = stream['reward']
    curves[f'{name}_cont'] = stream['cont']
  curves['policy_action_step'] = np.abs(np.diff(streams['policy']['action'][:-1], axis=0)).mean(-1)
  curves['policy_action_rmse_rad'] = np.sqrt(np.square(
      to_radians(streams['policy']['action'][:-1]) - to_radians(expert_actions[:-1])).mean(-1))
  # The policy's action on each posterior state (real obs, expert history), never executed.
  observed = posterior['policy_action']
  curves['posterior_policy_action_step'] = np.abs(np.diff(observed[:-1], axis=0)).mean(-1)
  curves['posterior_policy_action_rmse_rad'] = np.sqrt(np.square(
      to_radians(observed[:-1]) - to_radians(expert_actions[:-1])).mean(-1))
  return curves


def summarize(curves, episode, start, end, hz):
  def stats(name):
    tail = max(1, len(curves[f'{name}_force_rmse']) // 20)
    return {
        'latent_reward_mean': float(curves[f'{name}_latent_reward'].mean()),
        'force_rmse_counts': float(curves[f'{name}_force_rmse'].mean()),
        'position_error_mm': float(curves[f'{name}_position_error_mm'].mean()),
        'position_error_mm_last_5pct': float(curves[f'{name}_position_error_mm'][-tail:].mean()),
        'predicted_reward_last_5pct': float(curves[f'{name}_reward'][-tail:].mean()),
        'position_step_mm': float(curves[f'{name}_position_step_mm'].mean()),
    }

  return {
      'start_step': start,
      'end_step': end,
      'hz': hz,
      'imagined_seconds': (end - 1 - start) / hz if hz else None,
      'real_final_reward': float(episode['reward'][-1]),
      'policy': stats('policy'),
      'expert_actions': stats('expert'),
      'posterior': stats('posterior'),
      'policy_action_rmse_rad': float(curves['policy_action_rmse_rad'].mean()),
      'real_position_step_mm': float(curves['real_position_step_mm'].mean()),
      'policy_action_step': float(curves['policy_action_step'].mean()),
      'posterior_policy_action_step': float(curves['posterior_policy_action_step'].mean()),
      'posterior_policy_action_rmse_rad': float(curves['posterior_policy_action_rmse_rad'].mean()),
      'expert_action_step': float(curves['expert_action_step'].mean()),
  }


def plot_summary(path, title, episode, streams, curves, start, end, hz):
  t = (np.arange(end) / hz) if hz else np.arange(end)
  imagined = t[start:end]
  xlabel = 'Time (s)' if hz else 'Step'
  fig = plt.figure(figsize=(16, 15), constrained_layout=True)
  grid = fig.add_gridspec(6, 4)

  expert = to_radians(episode['action'][:end])
  observed = to_radians(streams['posterior']['policy_action'])
  policy = to_radians(streams['policy']['action'])
  for joint in range(16):
    ax = fig.add_subplot(grid[joint // 4, joint % 4])
    ax.plot(t[:end - 1], expert[:end - 1, joint], color='black', lw=1.2, label='Expert')
    ax.plot(t[:end - 1], observed[:end - 1, joint], color='tab:green', lw=0.8, alpha=0.8,
            label='Policy on real observations')
    ax.plot(imagined[:-1], policy[:-1, joint], color='tab:red', lw=0.8, label='Policy in world model')
    ax.axvline(t[start], color='gray', ls=':', lw=1)
    ax.set_title(f'Joint {joint} command (rad)', fontsize=9)
    ax.tick_params(labelsize=7)
    if joint == 0:
      ax.legend(fontsize=7)

  def panel(row, col, name, keys, ylabel, log=False):
    ax = fig.add_subplot(grid[row, col])
    for key, label, color in keys:
      series = curves[key]
      ax.plot(imagined[:len(series)], series, color=color, lw=1, label=label)
    if log:
      ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7)

  policy_label, policy_color = 'Prior, policy', STREAMS['policy'][1]
  expert_label, expert_color = 'Prior, expert actions', STREAMS['expert'][1]
  post_label, post_color = 'Posterior', STREAMS['posterior'][1]
  both = lambda suffix: [
      (f'posterior_{suffix}', post_label, post_color),
      (f'policy_{suffix}', policy_label, policy_color),
      (f'expert_{suffix}', expert_label, expert_color)]
  panel(4, 0, 'Mean taxel ||force||', [
      ('real_force_norm', 'Real', 'black'), *both('force_norm')], 'Counts')
  panel(4, 1, 'Force error vs real', both('force_rmse'), 'RMSE (counts)')
  panel(4, 2, 'Taxel position error vs real', both('position_error_mm'), 'mm')
  panel(4, 3, 'Latent reward (eq. 8)', both('latent_reward')[1:], 'Reward')
  panel(5, 0, 'Command change per step', [
      ('posterior_policy_action_step', 'Policy on posterior', post_color),
      ('policy_action_step', 'Policy in prior', policy_color),
      ('expert_action_step', 'Expert', 'black')], 'Mean |delta| (normalised)', log=True)
  panel(5, 1, 'Policy command error', [
      ('posterior_policy_action_rmse_rad', 'Policy on posterior', post_color),
      ('policy_action_rmse_rad', 'Policy in prior', policy_color)], 'RMSE (rad)')
  panel(5, 2, 'Predicted task reward', both('reward'), 'Reward head')
  panel(5, 3, 'Predicted continuation', both('cont'), 'Continue head')
  fig.suptitle(title, fontsize=12)
  fig.savefig(path, dpi=80)
  plt.close(fig)


def render_gif(path, title, episode, streams, curves, start, end, hz, fps, elev, azim):
  real_positions = episode['positions'][start:end].astype(np.float32)
  real_forces = episode['forces_local'][start:end].astype(np.float32)
  stride = max(1, round(hz / fps)) if hz else 1
  frames = range(0, end - start, stride)
  delay = int(round(1000 * stride / hz)) if hz else int(round(1000 / fps))
  posterior = streams['posterior']
  panels = {
      'real': (real_positions, real_forces),
      'posterior': (posterior['positions'][start:end], posterior['forces'][start:end]),
      'expert': (streams['expert']['positions'], streams['expert']['forces']),
      'policy': (streams['policy']['positions'], streams['policy']['forces']),
  }
  # One colour scale and one set of axes, so the panels can be compared directly.
  vmax = max(float(np.percentile(np.linalg.norm(real_forces, axis=-1), 99.5)), 1.0)
  flat = real_positions.reshape(-1, 3)
  low, high = np.percentile(flat, 0.1, axis=0), np.percentile(flat, 99.9, axis=0)
  center, radius = (low + high) / 2, float((high - low).max()) / 2 * 1.1

  fig = plt.figure(figsize=(18, 8.5), dpi=80)
  grid = fig.add_gridspec(2, 4, height_ratios=(2.2, 1), left=0.03, right=0.93, top=0.88,
                          bottom=0.07, wspace=0.05, hspace=0.18)
  scatters = {}
  for i, (name, (positions, forces)) in enumerate(panels.items()):
    ax = fig.add_subplot(grid[0, i], projection='3d')
    ax.view_init(elev=elev, azim=azim)
    for axis, mid in zip('xyz', center):
      getattr(ax, f'set_{axis}lim')(mid - radius, mid + radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.set_title(STREAMS[name][0], fontsize=11, color=STREAMS[name][1])
    scatters[name] = ax.scatter(
        *positions[0].T, c=np.linalg.norm(forces[0], axis=-1), s=10,
        cmap='magma', vmin=0, vmax=vmax, depthshade=False)
  bar = fig.add_axes([0.945, 0.45, 0.01, 0.4])
  fig.colorbar(scatters['real'], cax=bar, label='||force|| (counts)')
  heading = fig.suptitle(title, fontsize=11)

  # Jitter traces over the imagined window, with a cursor at the current frame.
  times = (np.arange(end - start) / hz) if hz else np.arange(end - start)
  xlabel = 'Time after start (s)' if hz else 'Step after start'

  def strip(slot, name, keys, ylabel, log=False):
    ax = fig.add_subplot(slot)
    for key, label, color in keys:
      series = curves[key]
      ax.plot(times[1:len(series) + 1], series, color=color, lw=1, label=label)
    if log:
      ax.set_yscale('symlog', linthresh=1e-4)
    ax.set_xlim(times[0], times[-1])
    ax.set_title(name, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=7, loc='upper right')
    return ax.axvline(times[0], color='gray', lw=1)

  cursors = [
      strip(grid[1, 0:2], 'Command change per step (jitter)', [
          ('expert_action_step', 'Expert', 'black'),
          ('posterior_policy_action_step', 'Policy on posterior', STREAMS['posterior'][1]),
          ('policy_action_step', 'Policy in prior', STREAMS['policy'][1])],
          'Mean |delta| (normalised)', log=True),
      strip(grid[1, 2:4], 'Decoded taxel movement per step (jitter)', [
          ('real_position_step_mm', 'Real', 'black'),
          *[(f'{name}_position_step_mm', STREAMS[name][0], STREAMS[name][1])
            for name in DECODED]], 'Mean |delta| (mm)'),
  ]

  images = []
  for frame in frames:
    for name, (positions, forces) in panels.items():
      scatters[name]._offsets3d = tuple(positions[frame].T)
      scatters[name].set_array(np.linalg.norm(forces[frame], axis=-1))
    for cursor in cursors:
      cursor.set_xdata([times[frame], times[frame]])
    clock = f'{(frame / hz):.2f} s after start' if hz else f'step {frame}'
    heading.set_text(f'{title}\n{clock}')
    fig.canvas.draw()
    rgb = Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
    images.append(rgb.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))
  plt.close(fig)
  images[0].save(
      path, save_all=True, append_images=images[1:], duration=delay, loop=0, optimize=False)
  return len(images)


def data_dir_for(config):
  _, task = config.task.split('_', 1)
  return _REPO_ROOT / 'data' / 'leap' / task.split('-', 1)[0]


def data_rate(data_dir, override):
  if override is not None:
    return override
  try:
    return float(json.loads((data_dir / 'manifest.json').read_text())['freq'])
  except (OSError, ValueError, KeyError, TypeError):
    return None


def load_episode(path):
  with np.load(path) as raw:
    return {key: raw[key] for key in raw.files}


def final_reward(path):
  with np.load(path) as raw:
    return float(raw['reward'][-1])


def choose_episodes(args, data_dir, logdir, config):
  """Paths for --episode, or the first --num-episodes matching --split and --outcome."""
  if args.episode is not None and not args.episode.isdigit():
    path = data_dir / args.episode
    if not path.exists():
      raise SystemExit(f'No episode {path}')
    return [path]
  paths = sorted(data_dir.glob('*.npz'))
  if not paths:
    raise SystemExit(f'No .npz episodes in {data_dir}')
  val_file = logdir / 'val_files.txt'
  if args.split != 'all':
    if not val_file.exists():
      raise SystemExit(f'{val_file} is missing. Use --split all.')
    held_out = set(val_file.read_text().split())
    paths = [p for p in paths if (p.name in held_out) == (args.split == 'val')]
  threshold = float(config.ditto_expert_min_reward)
  if args.outcome == 'success':
    paths = [p for p in paths if final_reward(p) >= threshold]
  elif args.outcome == 'fail':
    paths = [p for p in paths if final_reward(p) < threshold]
  if not paths:
    raise SystemExit(f'No episodes for split={args.split} outcome={args.outcome}.')
  print(f'{len(paths)} episodes match split={args.split} outcome={args.outcome}')
  first = int(args.episode) if args.episode is not None else 0
  if first >= len(paths):
    raise SystemExit(f'--episode {first} is out of range ({len(paths)} episodes match).')
  return paths[first:first + args.num_episodes]


def main():
  parser = argparse.ArgumentParser(
      description='Run a trained policy inside its world model and decode the result.')
  parser.add_argument(
      '--logdir', type=Path, default=Path('logdir/ditto/leap_mechanical_pen/0'),
      help='Run folder with config.yaml and checkpoint.ckpt')
  parser.add_argument(
      '--data', type=Path, default=None,
      help='Episode folder (default: data/leap/<task> from the run config)')
  parser.add_argument(
      '--episode', default=None, help='File name, or index into the matching episodes (default: 0)')
  parser.add_argument('--num-episodes', type=int, default=1)
  parser.add_argument(
      '--split', choices=('val', 'train', 'all'), default='val',
      help='val is hidden from the policy, but the world model trained on all episodes')
  parser.add_argument('--outcome', choices=('success', 'fail', 'any'), default='success')
  parser.add_argument(
      '--start', type=int, default=8, help='Last real step observed before imagining (default: 8)')
  parser.add_argument('--steps', type=int, default=None, help='Imagined steps (default: to the end)')
  parser.add_argument('--hz', type=float, default=None, help='Default: freq in the data manifest')
  parser.add_argument('--fps', type=float, default=10.0, help='GIF frame rate; plays in real time')
  parser.add_argument('--elev', type=float, default=25.0, help='GIF camera elevation')
  parser.add_argument('--azim', type=float, default=-60.0, help='GIF camera azimuth')
  parser.add_argument('--no-gif', action='store_true')
  parser.add_argument('--out', type=Path, default=None, help='Default: <logdir>/rollout_wm')
  args = parser.parse_args()

  logdir = args.logdir.expanduser().resolve()
  agent, config = load_agent(logdir)
  data_dir = (args.data or data_dir_for(config)).expanduser().resolve()
  hz = data_rate(data_dir, args.hz)
  if hz is None:
    print('Warning: no rate found, so time axes use steps and the GIF speed is not real time. '
          'Pass --hz to fix this.', file=sys.stderr)
  paths = choose_episodes(args, data_dir, logdir, config)
  model = WorldModelRollout(agent, config)

  print(f'Algo: {config.algo}  Task: {config.task}  Rate: {hz} Hz  Episodes: {len(paths)}')
  for path in paths:
    episode = load_episode(path)
    length = len(episode['action'])
    if not 0 <= args.start < length - 1:
      raise SystemExit(f'--start {args.start} is outside episode of {length} steps ({path.name}).')
    end = length if args.steps is None else min(length, args.start + args.steps + 1)
    print(f'\n{path.name}: observing to step {args.start}, imagining to step {end - 1}')
    streams = run_episode(model, episode, args.start, end)
    curves = compute_curves(episode, streams, args.start, end)
    metrics = summarize(curves, episode, args.start, end, hz)
    metrics.update(episode=path.name, algo=config.algo, logdir=str(logdir))

    out = (args.out.expanduser().resolve() if args.out else logdir / 'rollout_wm') / path.stem
    out.mkdir(parents=True, exist_ok=True)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=1))
    np.savez_compressed(
        out / 'rollout.npz',
        **curves,
        real_positions=episode['positions'][args.start:end],
        real_forces=episode['forces_local'][args.start:end],
        observed_policy_action=streams['posterior']['policy_action'][args.start:end],
        posterior_positions=streams['posterior']['positions'][args.start:end],
        posterior_forces=streams['posterior']['forces'][args.start:end],
        **{f'{name}_{key}': streams[name][key]
           for name in ('policy', 'expert')
           for key in ('positions', 'forces', 'action')})
    label = f'{config.algo} in world model, final reward {float(episode["reward"][-1]):+.0f}'
    plot_summary(out / 'summary.png', f'{path.name}\n{label}', episode, streams, curves,
                 args.start, end, hz)
    if not args.no_gif:
      count = render_gif(out / 'rollout.gif', label, episode, streams, curves, args.start, end, hz,
                         args.fps, args.elev, args.azim)
      print(f'  GIF: {count} frames')

    policy, expert, post = metrics['policy'], metrics['expert_actions'], metrics['posterior']
    print(f'  {"":24}{"Prior policy":>14}{"Prior expert":>14}{"Posterior":>12}')
    for key, name in (
        ('latent_reward_mean', 'Latent reward'),
        ('force_rmse_counts', 'Force RMSE (counts)'),
        ('position_error_mm', 'Position error (mm)'),
        ('position_error_mm_last_5pct', 'Same, last 5%'),
        ('predicted_reward_last_5pct', 'Pred. reward, last 5%'),
        ('position_step_mm', 'Taxel move/step (mm)')):
      print(f'  {name:<24}{policy[key]:>14.3f}{expert[key]:>14.3f}{post[key]:>12.3f}')
    print(f'  Real taxel move/step: {metrics["real_position_step_mm"]:.3f} mm')
    print(f'  Command error: policy in prior {metrics["policy_action_rmse_rad"]:.3f}, '
          f'policy on posterior {metrics["posterior_policy_action_rmse_rad"]:.3f} rad RMS')
    print(f'  Command change per step: policy in prior {metrics["policy_action_step"]:.5f}, '
          f'policy on posterior {metrics["posterior_policy_action_step"]:.5f}, '
          f'expert {metrics["expert_action_step"]:.5f}')
    print(f'  Saved to {out}')


if __name__ == '__main__':
  main()

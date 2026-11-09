#!/usr/bin/env python3
"""Gradio visualizer for trained Leap DreamerV3 world models.

Browse offline demo trajectories, pick a timestep, and inspect teacher-forced
reconstructions plus open-loop imagination predictions.

Example:
  python eval/mechanical_pen/visualize_model.py \\
      --logdir logdir/model/leap_mechanical_pen/0/0 \\
      --data data/leap/mechanical_pen
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import gradio as gr
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
  sys.path.insert(0, str(_REPO_ROOT))

import embodied  # noqa: E402
from dreamerv3 import ninjax as nj  # noqa: E402
from dreamerv3.main import make_agent  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_density import SPACES, DataDensity  # noqa: E402


def _to_numpy(tree):
  with jax._src.config.explicit_device_get_scope():
    host = jax.device_get(tree)

  def convert(x):
    x = np.asarray(x)
    if x.ndim > 0 and x.shape[0] == 1:
      return np.array(x[0])
    return x

  return jax.tree_util.tree_map(convert, host)


def load_episode(path: Path) -> dict[str, np.ndarray]:
  raw = np.load(path)
  return {k: np.asarray(raw[k]) for k in raw.files if not k.startswith('stepid')}


def _pad_time(value: np.ndarray, length: int) -> np.ndarray:
  if value.shape[0] >= length:
    return value[:length]
  out = np.zeros((length, *value.shape[1:]), value.dtype)
  out[:value.shape[0]] = value
  return out


def slice_window(episode: dict[str, np.ndarray], start: int, length: int):
  end = min(start + length, len(episode['action']))
  actual = end - start
  batch = {}
  for key, value in episode.items():
    if value.ndim == 0:
      batch[key] = value
      continue
    chunk = value[start:end]
    if chunk.ndim == 1:
      chunk = _pad_time(chunk, length)
    else:
      chunk = _pad_time(chunk, length)
    batch[key] = chunk[None]
  batch['is_first'] = np.zeros((1, length), bool)
  batch['is_first'][0, 0] = True
  batch['is_last'] = np.zeros((1, length), bool)
  if actual > 0:
    batch['is_last'][0, actual - 1] = True
  if 'is_terminal' in batch:
    term = np.zeros((1, length), bool)
    if actual > 0:
      term[0, actual - 1] = bool(episode['is_terminal'][end - 1])
    batch['is_terminal'] = term
  return batch, actual


def taxel_magnitude(field: np.ndarray) -> np.ndarray:
  return np.linalg.norm(field, axis=-1)


def scatter_taxels(
    ax,
    positions: np.ndarray,
    values: np.ndarray,
    title: str,
    cmap: str = 'magma',
    vmin: float | None = None,
    vmax: float | None = None,
):
  xy = positions[..., :2]
  sc = ax.scatter(
      xy[:, 0], xy[:, 1], c=values, s=8, cmap=cmap, vmin=vmin, vmax=vmax)
  ax.set_title(title, fontsize=10)
  ax.set_aspect('equal', adjustable='box')
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  return sc


def figure_to_image(fig) -> np.ndarray:
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
  return buf[..., :3].copy()


class LeapModelVisualizer:

  def __init__(self, logdir: Path, data_dir: Path):
    self.logdir = logdir.expanduser().resolve()
    self.data_dir = data_dir.expanduser().resolve()
    self.window = 64
    self.agent, self.config = self._load_agent()
    self.episodes = sorted(self.data_dir.glob('*.npz'))
    if not self.episodes:
      raise FileNotFoundError(f'No .npz episodes in {self.data_dir}')
    self._visualize = self._compile_visualize()
    self.density = DataDensity(self.episodes)

  def _load_agent(self):
    config_path = self.logdir / 'config.yaml'
    if not config_path.exists():
      raise FileNotFoundError(f'Missing config: {config_path}')
    from ruamel.yaml import YAML
    yaml = YAML(typ='safe')
    config = embodied.Config(yaml.load(config_path.read_text()))
    config = config.update(logdir=str(self.logdir))
    agent = make_agent(config)
    checkpoint = embodied.Checkpoint(self.logdir / 'checkpoint.ckpt')
    checkpoint.agent = agent
    checkpoint.load(keys=['agent'])
    return agent, config

  def _compile_visualize(self):
    inner = self.agent.agent
    tm = self.agent.train_mirrored
    ts = self.agent.train_sharded

    def visualize(params, data, carry, seed):
      pure = nj.pure(inner.visualize)
      _, (outs, carry) = pure(params, data, carry, seed=seed, create=True)
      return outs, carry

    fn = jax.jit(visualize, (tm, ts, ts, ts), (ts, ts))
    episode = load_episode(self.episodes[0])
    batch, _ = slice_window(episode, 0, self.window)
    seed = self.agent._next_seeds(ts)
    carry = self.agent.init_report(1)
    data = jax.device_put(self.agent._filter_data(batch), ts)
    fn(self.agent.params, data, carry, seed)
    return fn

  @lru_cache(maxsize=8)
  def _infer_cached(self, episode_name: str, start: int):
    episode = load_episode(self.data_dir / episode_name)
    batch, actual = slice_window(episode, start, self.window)
    length = actual
    seed = self.agent._next_seeds(self.agent.train_sharded)
    carry = self.agent.init_report(1)
    data = jax.device_put(
        self.agent._filter_data(batch), self.agent.train_sharded)
    outs, _ = self._visualize(self.agent.params, data, carry, seed)
    return _to_numpy(outs), length

  def episode_choices(self) -> list[str]:
    return [p.name for p in self.episodes]

  def episode_length(self, episode_name: str) -> int:
    return len(load_episode(self.data_dir / episode_name)['action'])

  def render(
      self,
      episode_name: str,
      window_start: int,
      step: int,
      view: str,
      density_space: str = SPACES[0],
      density_radius: float = 0.1,
  ):
    outs, length = self._infer_cached(episode_name, int(window_start))
    step = int(np.clip(step, 0, length - 1))
    num_obs = min(int(self.config.report_openl_context), length // 2)

    if view.startswith('Forces'):
      gt = outs['forces_local_gt'][step]
      recon = outs['forces_local_recon'][step]
      openl = outs['forces_local_openl'][step]
      gt_mag = taxel_magnitude(gt)
      recon_mag = taxel_magnitude(recon)
      openl_mag = taxel_magnitude(openl)
      pos_layout = outs['positions_gt'][step]
      mse_recon = float(outs['forces_mse_recon'][step])
      mse_openl = float(outs['forces_mse_openl'][step])
      ylabel = '||force||'
    else:
      gt = outs['positions_gt'][step]
      recon = outs['positions_recon'][step]
      openl = outs['positions_openl'][step]
      gt_mag = taxel_magnitude(gt)
      recon_mag = taxel_magnitude(recon)
      openl_mag = taxel_magnitude(openl)
      pos_layout = gt
      mse_recon = float(outs['positions_mse_recon'][step])
      mse_openl = float(outs['positions_mse_openl'][step])
      ylabel = '||position||'

    vmax = max(gt_mag.max(), recon_mag.max(), np.nanmax(openl_mag), 1e-8)
    err_recon = np.abs(recon_mag - gt_mag)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    sc0 = scatter_taxels(
        axes[0, 0], pos_layout, gt_mag, 'Ground truth', vmax=vmax)
    scatter_taxels(
        axes[0, 1], pos_layout, recon_mag, 'Teacher-forced recon', vmax=vmax)
    if step >= num_obs and not np.isnan(openl_mag).all():
      scatter_taxels(
          axes[0, 2], pos_layout, openl_mag, 'Open-loop imagine', vmax=vmax)
    else:
      axes[0, 2].text(
          0.5, 0.5,
          f'Open-loop N/A\n(needs step >= {num_obs})',
          ha='center', va='center', transform=axes[0, 2].transAxes)
      axes[0, 2].set_axis_off()
    scatter_taxels(
        axes[1, 0], pos_layout, err_recon, '|recon - gt|', cmap='coolwarm')
    if step >= num_obs and not np.isnan(openl_mag).all():
      scatter_taxels(
          axes[1, 1], pos_layout, np.abs(openl_mag - gt_mag),
          '|openl - gt|', cmap='coolwarm')
    else:
      axes[1, 1].set_axis_off()
    fig.colorbar(sc0, ax=axes.ravel().tolist(), shrink=0.75, label=ylabel)

    t = np.arange(length)
    axes[1, 2].plot(t, outs['forces_mse_recon'], label='forces recon mse')
    axes[1, 2].plot(t, outs['positions_mse_recon'], label='positions recon mse')
    if not np.all(np.isnan(outs['forces_mse_openl'])):
      axes[1, 2].plot(
          t, outs['forces_mse_openl'], '--', label='forces openl mse')
      axes[1, 2].plot(
          t, outs['positions_mse_openl'], '--', label='positions openl mse')
    axes[1, 2].axvline(step, color='black', linestyle=':', linewidth=1.5)
    axes[1, 2].set_xlabel('Step in window')
    axes[1, 2].set_ylabel('MSE')
    axes[1, 2].set_title('Sequence error')
    axes[1, 2].legend(fontsize=8)

    fig.suptitle(
        f'{episode_name}  window=[{window_start}, {window_start + length})  '
        f'step={step}  openl_context={num_obs}',
        fontsize=11)

    image = figure_to_image(fig)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)
    axes2[0].plot(t, outs['reward_gt'], label='reward gt', color='black')
    axes2[0].plot(t, outs['reward_recon'], label='reward recon')
    axes2[0].plot(t, outs['reward_openl'], '--', label='reward openl')
    axes2[0].axvline(step, color='gray', linestyle=':')
    axes2[0].set_title('Reward')
    axes2[0].legend(fontsize=8)

    axes2[1].plot(t, outs['cont_gt'], label='cont gt', color='black')
    axes2[1].plot(t, outs['cont_recon'], label='cont recon')
    axes2[1].plot(t, outs['cont_openl'], '--', label='cont openl')
    axes2[1].axvline(step, color='gray', linestyle=':')
    axes2[1].set_title('Continuation')
    axes2[1].legend(fontsize=8)

    scalars = figure_to_image(fig2)
    plt.close(fig2)

    heads = self.render_heads(outs, view, step, length, num_obs, pos_layout)

    openl_note = (
        f'{mse_openl:.6g}' if step >= num_obs and not np.isnan(mse_openl)
        else 'n/a (step < open-loop context)')
    metrics = (
        f'Episode: {episode_name}\n'
        f'Window start: {window_start}  length: {length}  step: {step}\n'
        f'Open-loop context: {num_obs} steps\n\n'
        f'At selected step:\n'
        f'  recon MSE (forces):    {mse_recon:.6g}\n'
        f'  openl MSE (forces):    {openl_note}\n'
        f'  recon MSE (positions): {float(outs["positions_mse_recon"][step]):.6g}\n'
        f'  reward gt / recon:     {outs["reward_gt"][step]:.4f} / '
        f'{outs["reward_recon"][step]:.4f}\n'
        f'  cont gt / recon:       {outs["cont_gt"][step]:.4f} / '
        f'{outs["cont_recon"][step]:.4f}\n'
    )
    density, density_text = self.render_density(
        episode_name, int(window_start), length, step, density_space,
        density_radius)
    return image, scalars, heads, density, metrics + '\n' + density_text

  def render_density(self, episode_name, start, length, step, space, radius):
    fig = plt.figure(figsize=(16, 4.8), constrained_layout=True)
    text = self.density.plot(fig, space, episode_name, start, length, step,
                             float(radius))
    image = figure_to_image(fig)
    plt.close(fig)
    return image, text

  def render_heads(self, outs, view, step, length, num_obs, pos_layout):
    """Open loop of every ensemble prior held fixed, plus per-prior MSE."""
    field = 'forces_local' if view.startswith('Forces') else 'positions'
    mse_key = 'forces' if field == 'forces_local' else 'positions'
    if f'{field}_openl_heads' not in outs:
      fig, ax = plt.subplots(figsize=(6, 1.5))
      ax.text(0.5, 0.5, 'World model has a single prior (ensemble=1)',
              ha='center', va='center')
      ax.set_axis_off()
      image = figure_to_image(fig)
      plt.close(fig)
      return image

    gt = outs[f'{field}_gt']
    heads = outs[f'{field}_openl_heads']
    mse = outs[f'{mse_key}_mse_openl_heads']
    num_heads = heads.shape[0]
    t = np.arange(length)
    colors = plt.get_cmap('tab10')(np.arange(num_heads) % 10)

    ncols = 4
    nrows = int(np.ceil((num_heads + 1) / ncols))
    fig = plt.figure(figsize=(14, 2.2 * nrows + 3.6), constrained_layout=True)
    grid = fig.add_gridspec(nrows + 1, ncols)

    gt_mag = taxel_magnitude(gt[step])
    head_mag = taxel_magnitude(heads[:, step])
    openl_valid = step >= num_obs
    vmax = max(gt_mag.max(), np.nanmax(head_mag) if openl_valid else 0, 1e-8)
    sc = scatter_taxels(
        fig.add_subplot(grid[0, 0]), pos_layout, gt_mag, 'Ground truth',
        vmax=vmax)
    for j in range(num_heads):
      ax = fig.add_subplot(grid[(j + 1) // ncols, (j + 1) % ncols])
      if openl_valid:
        scatter_taxels(
            ax, pos_layout, head_mag[j],
            f'Prior {j}  mse={mse[j, step]:.4g}', vmax=vmax)
      else:
        ax.text(0.5, 0.5, f'Prior {j}\nopen loop starts at {num_obs}',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_axis_off()
    fig.colorbar(sc, ax=fig.axes, shrink=0.6, label=f'||{field}||')

    ax_mse = fig.add_subplot(grid[nrows, :ncols // 2])
    for j in range(num_heads):
      ax_mse.plot(t, mse[j], color=colors[j], label=f'prior {j}')
    ax_mse.plot(t, outs[f'{mse_key}_mse_openl'], 'k--', label='sampled per step')
    ax_mse.plot(t, np.nanmean(mse, 0), 'k', linewidth=2, label='mean of priors')
    ax_mse.axvline(step, color='gray', linestyle=':')
    ax_mse.set_xlabel('Step in window')
    ax_mse.set_ylabel('MSE')
    ax_mse.set_title(f'Open-loop {mse_key} MSE per prior')
    ax_mse.legend(fontsize=7, ncol=2)

    ax_mag = fig.add_subplot(grid[nrows, ncols // 2:])
    ax_mag.plot(t, taxel_magnitude(gt).mean(-1), 'k', linewidth=2, label='gt')
    head_curve = taxel_magnitude(heads).mean(-1)
    for j in range(num_heads):
      ax_mag.plot(t, head_curve[j], color=colors[j], label=f'prior {j}')
    ax_mag.fill_between(
        t, np.nanmin(head_curve, 0), np.nanmax(head_curve, 0),
        color='gray', alpha=0.2, label='prior spread')
    ax_mag.axvline(num_obs, color='red', linestyle='--', linewidth=1,
                   label='open loop start')
    ax_mag.axvline(step, color='gray', linestyle=':')
    ax_mag.set_xlabel('Step in window')
    ax_mag.set_ylabel(f'mean taxel ||{field}||')
    ax_mag.set_title('Open-loop trajectory per prior')
    ax_mag.legend(fontsize=7, ncol=2)

    image = figure_to_image(fig)
    plt.close(fig)
    return image


def build_app(viz: LeapModelVisualizer) -> gr.Blocks:
  default_ep = viz.episode_choices()[0]
  default_len = viz.episode_length(default_ep)
  max_start = max(0, default_len - viz.window)

  with gr.Blocks(title='Leap World Model Visualizer') as demo:
    gr.Markdown(
        '# Leap DreamerV3 world model visualizer\n'
        'Inspect teacher-forced reconstructions and open-loop imagination '
        'on offline demo trajectories.')

    with gr.Row():
      logdir = gr.Textbox(value=str(viz.logdir), label='Checkpoint logdir')
      data_dir = gr.Textbox(value=str(viz.data_dir), label='Episode data dir')

    with gr.Row():
      episode = gr.Dropdown(
          choices=viz.episode_choices(), value=default_ep, label='Episode')
      view = gr.Radio(
          ['Forces', 'Positions'], value='Forces', label='Field')

    with gr.Row():
      window_start = gr.Slider(
          0, max_start, value=0, step=1, label='Window start')
      step = gr.Slider(
          0, viz.window - 1, value=0, step=1, label='Step in window')

    run_btn = gr.Button('Run inference', variant='primary')
    spatial = gr.Image(label='Spatial predictions')
    scalars = gr.Image(label='Reward & continuation')
    with gr.Row():
      density_space = gr.Radio(
          list(SPACES), value=SPACES[0], label='Density feature space')
      density_radius = gr.Slider(
          0.01, 1.0, value=0.1, step=0.01,
          label='Neighbourhood radius (x dataset spread)')
    density = gr.Image(label='Data density around current step')
    heads = gr.Image(label='Open loop per ensemble prior')
    metrics = gr.Textbox(label='Metrics', lines=18)

    def on_episode_change(name):
      length = viz.episode_length(name)
      max_s = max(0, length - viz.window)
      return gr.update(maximum=max_s, value=0)

    episode.change(on_episode_change, episode, window_start)

    def on_run(ep_name, start, st, field, d_space, d_radius, _logdir,
               _data_dir):
      viz._infer_cached.cache_clear()
      return viz.render(ep_name, int(start), int(st), field, d_space, d_radius)

    inputs = [episode, window_start, step, view, density_space,
              density_radius, logdir, data_dir]
    outputs = [spatial, scalars, heads, density, metrics]
    run_btn.click(
        on_run,
        inputs, outputs)
    for ctrl in (episode, window_start, step, view, density_space,
                 density_radius):
      ctrl.change(on_run, inputs, outputs)

  return demo


def main():
  parser = argparse.ArgumentParser(description='Leap world model Gradio visualizer')
  parser.add_argument(
      '--logdir',
      type=Path,
      default=_REPO_ROOT / 'logdir/model/leap_mechanical_pen/0/0',
      help='Training logdir containing checkpoint.ckpt and config.yaml')
  parser.add_argument(
      '--data',
      type=Path,
      default=_REPO_ROOT / 'data/leap/mechanical_pen',
      help='Directory of converted DreamerV3 .npz episodes')
  parser.add_argument('--host', default='0.0.0.0')
  parser.add_argument('--port', type=int, default=7860)
  parser.add_argument('--share', action='store_true')
  args = parser.parse_args()

  viz = LeapModelVisualizer(args.logdir, args.data)
  demo = build_app(viz)
  # Gradio only retries a range when server_port is None; a fixed port fails
  # hard if something else (often a previous visualizer) still holds it.
  try:
    demo.launch(
        server_name=args.host, server_port=args.port, share=args.share)
  except OSError as exc:
    if 'Cannot find empty port' not in str(exc):
      raise
    print(f'Port {args.port} busy; picking a free port…')
    demo.launch(server_name=args.host, server_port=None, share=args.share)


if __name__ == '__main__':
  main()

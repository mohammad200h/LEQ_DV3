#!/usr/bin/env python3
"""Gradio explorer for the mechanical pen training data (no model needed).

Lets you look at the demo episodes the world model is trained on:

  Overview        how many episodes, how long, success/fail, train/val split
  Episode         one episode frame by frame: taxel forces on the hand, contact
                  timeline per sensor patch, joint targets, reward
  Tactile stats   force/position value distributions (raw, z-scored, symlog),
                  which taxels are ever touched, which ones drop out
  Actions         per-joint target distributions and joint correlations
  Success vs fail per-finger force aligned to the episode end
  Coverage        the PCA "map" of all timesteps (shared with the model
                  visualizer's density view), coloured by outcome and split

Dataset-wide statistics are computed in one streaming pass and cached under
eval/mechanical_pen/.cache/. The course in eval/mechanical_pen/course/ walks
through every plot.

Example:
  python eval/mechanical_pen/explore_data.py --data data/leap/mechanical_pen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import gradio as gr
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
  sys.path.insert(0, str(_HERE))
_REPO_ROOT = _HERE.parents[1]
_CACHE_DIR = _HERE / '.cache'
_CACHE_VERSION = 1

# ---------------------------------------------------------------- layout

# Taxel order of forces_local / positions (Sparsh XELA_FLATTEN_ORDER):
# (patch name, finger, number of taxels). 4 fingertips x 30 + 11 phalanx
# patches x 16 + 3 palm patches x 24 = 368.
PATCHES = [
    ('thumb tip', 'thumb', 30), ('thumb link15', 'thumb', 16),
    ('thumb link14', 'thumb', 16),
    ('index tip', 'index', 30), ('index link2', 'index', 16),
    ('index link1A', 'index', 16), ('index link1B', 'index', 16),
    ('middle tip', 'middle', 30), ('middle link6', 'middle', 16),
    ('middle link5A', 'middle', 16), ('middle link5B', 'middle', 16),
    ('ring tip', 'ring', 30), ('ring link10', 'ring', 16),
    ('ring link9A', 'ring', 16), ('ring link9B', 'ring', 16),
    ('palm 2', 'palm', 24), ('palm 1', 'palm', 24), ('palm 3', 'palm', 24),
]
FINGERS = ('thumb', 'index', 'middle', 'ring', 'palm')
FINGER_COLORS = {'thumb': 'tab:red', 'index': 'tab:blue', 'middle': 'tab:green',
                 'ring': 'tab:purple', 'palm': 'tab:gray'}
N_TAXELS = sum(n for _, _, n in PATCHES)
_bounds = np.cumsum([0] + [n for _, _, n in PATCHES])
PATCH_SLICES = [slice(a, b) for a, b in zip(_bounds[:-1], _bounds[1:])]
TAXEL_PATCH = np.repeat(np.arange(len(PATCHES)), [n for _, _, n in PATCHES])
TAXEL_FINGER = np.array([FINGERS.index(PATCHES[p][1]) for p in TAXEL_PATCH])

# Action columns (sim joint order, from the episodes' metadata joint_order).
JOINTS = ('if_mcp', 'if_rot', 'if_pip', 'if_dip',
          'mf_mcp', 'mf_rot', 'mf_pip', 'mf_dip',
          'rf_mcp', 'rf_rot', 'rf_pip', 'rf_dip',
          'th_cmc', 'th_axl', 'th_mcp', 'th_ipl')
JOINT_FINGER = ('index',) * 4 + ('middle',) * 4 + ('ring',) * 4 + ('thumb',) * 4
ACTION_LOW, ACTION_HIGH = -1.5, 2.5   # rad, scaled to [-1, 1] in the chunks

# forces_norm in configs.yaml (mechanical_pen); the world model sees
# symlog((forces - mean) / std).
FORCE_MEAN = np.array([6.1331, 8.4556, 28.8512], np.float32)
FORCE_STD = np.array([140.4322, 94.8270, 145.6174], np.float32)

WINDOW = 64          # batch_length_eval; training uses 65-step windows
OPENL_CONTEXT = 8    # report_openl_context
ALIGN = 400          # steps kept at the start / end of every episode
FREQ = 100.0         # Hz

# Histogram bins for the dataset statistics.
B_FRAW = np.linspace(-1500, 1500, 151)
B_FZ = np.linspace(-12, 12, 121)
B_FSYM = np.linspace(-3.5, 3.5, 141)
B_POS = np.linspace(-0.15, 0.3, 181)
B_ACT = np.linspace(-1, 1, 81)
B_MAG = np.concatenate([[0], np.logspace(0, 4, 41)])   # per-taxel |force|
B_DACT = np.concatenate([[0], np.logspace(-5, 0, 51)])  # |a[t+1]-a[t]|


def symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))


def load_episode(path: Path) -> dict[str, np.ndarray]:
  raw = np.load(path)
  return {k: np.asarray(raw[k]) for k in raw.files if not k.startswith('stepid')}


def figure_to_image(fig) -> np.ndarray:
  fig.canvas.draw()
  buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
  return buf[..., :3].copy()


def finger_curves(mag: np.ndarray) -> np.ndarray:
  """(T, 368) taxel |force| -> (T, 5) mean |force| per finger."""
  return np.stack([mag[:, TAXEL_FINGER == i].mean(1)
                   for i in range(len(FINGERS))], 1)


def patch_curves(mag: np.ndarray) -> np.ndarray:
  """(T, 368) taxel |force| -> (T, 18) mean |force| per sensor patch."""
  return np.stack([mag[:, s].mean(1) for s in PATCH_SLICES], 1)


def split_demos(names, val_frac=0.1, seed=0):
  """Same split as dreamerv3/main.py split_demos (sorted names, fixed seed)."""
  names = sorted(names)
  if val_frac <= 0:
    return set()
  perm = np.random.default_rng(seed).permutation(len(names))
  nval = max(1, int(round(len(names) * val_frac)))
  return {names[i] for i in perm[:nval]}


# ---------------------------------------------------------------- stats

def _episode_stats(path: Path) -> dict:
  ep = load_episode(path)
  f = ep['forces_local'].astype(np.float32)
  p = ep['positions'].astype(np.float32)
  a = ep['action'].astype(np.float32)
  r = ep['reward'].astype(np.float32)
  t = len(a)
  mag = np.linalg.norm(f, axis=-1)
  fz = (f - FORCE_MEAN) / FORCE_STD
  fing = finger_curves(mag)

  def aligned(x, at_end):
    out = np.full((ALIGN, x.shape[1]), np.nan, np.float32)
    n = min(ALIGN, len(x))
    if at_end:
      out[ALIGN - n:] = x[len(x) - n:]
    else:
      out[:n] = x[:n]
    return out

  s = {
      'length': t,
      'final_reward': float(r[-1]),
      'reward_sum': float(r.sum()),
      'nonzero_rewards': int((r != 0).sum()),
      'terminal': bool(ep['is_terminal'][-1]),
      'max_force': float(mag.max()),
      'mean_force_finger': fing.mean(0),
      'nonfinite': int((~np.isfinite(f)).sum() + (~np.isfinite(p)).sum()
                       + (~np.isfinite(a)).sum()),
      'end_finger': aligned(fing, True),
      'start_finger': aligned(fing, False),
      'action_mean': a.mean(0),
      'pos_motion': float(np.sqrt((np.diff(p, axis=0) ** 2).sum(-1)).mean()),
      'h_fraw': np.stack([np.histogram(f[..., k], B_FRAW)[0] for k in range(3)]),
      'h_fz': np.stack([np.histogram(fz[..., k], B_FZ)[0] for k in range(3)]),
      'h_fsym': np.stack([np.histogram(symlog(fz[..., k]), B_FSYM)[0]
                          for k in range(3)]),
      'h_fsym_raw': np.stack([np.histogram(symlog(f[..., k]), B_FSYM * 3)[0]
                              for k in range(3)]),
      'h_pos': np.stack([np.histogram(p[..., k], B_POS)[0] for k in range(3)]),
      'h_act': np.stack([np.histogram(a[:, j], B_ACT)[0] for j in range(16)]),
      'h_dact': np.stack([np.histogram(np.abs(np.diff(a[:, j])), B_DACT)[0]
                          for j in range(16)]),
      'h_mag_taxel': np.stack([np.histogram(mag[:, i], B_MAG)[0]
                               for i in range(N_TAXELS)]),
      'zero_taxel': (f == 0).all(-1).sum(0),
      'f_sum': f.reshape(-1, 3).astype(np.float64).sum(0),
      'f_sq': (f.reshape(-1, 3).astype(np.float64) ** 2).sum(0),
      'p_sum': p.reshape(-1, 3).astype(np.float64).sum(0),
      'p_sq': (p.reshape(-1, 3).astype(np.float64) ** 2).sum(0),
      'p_tvar': p.var(0).mean(),          # per-dim variance over time
      'fz_tvar': symlog(fz).var(0).mean(),
      'a_sum': a.astype(np.float64).sum(0),
      'a_outer': a.astype(np.float64).T @ a.astype(np.float64),
      'pos_layout': p[0],
  }
  return s


class DatasetStats:
  """Per-episode summaries plus pooled histograms, cached on disk."""

  SUM_KEYS = ('h_fraw', 'h_fz', 'h_fsym', 'h_fsym_raw', 'h_pos', 'h_act',
              'h_dact', 'h_mag_taxel', 'zero_taxel', 'f_sum', 'f_sq', 'p_sum',
              'p_sq', 'a_sum', 'a_outer')
  EP_KEYS = ('length', 'final_reward', 'reward_sum', 'nonzero_rewards',
             'terminal', 'max_force', 'mean_force_finger', 'nonfinite',
             'end_finger', 'start_finger', 'action_mean', 'pos_motion',
             'p_tvar', 'fz_tvar')

  def __init__(self, episodes: list[Path]):
    self.episodes = episodes
    path = self._cache_path()
    if path.exists():
      data = np.load(path)
      self.d = {k: data[k] for k in data.files}
      print(f'[explore] loaded cached dataset stats {path}')
      return
    print(f'[explore] computing statistics of {len(episodes)} episodes '
          '(one-off, about a minute; cached afterwards)...')
    self.d = self._build()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, **self.d)
    print(f'[explore] cached to {path}')

  def _cache_path(self) -> Path:
    h = hashlib.sha1()
    h.update(f'explore{_CACHE_VERSION}'.encode())
    for p in self.episodes:
      h.update(p.name.encode())
    return _CACHE_DIR / f'explore_{h.hexdigest()[:16]}.npz'

  def _build(self) -> dict:
    sums, per_ep = {}, {k: [] for k in self.EP_KEYS}
    layout = None
    with ThreadPoolExecutor(16) as pool:
      for i, s in enumerate(pool.map(_episode_stats, self.episodes)):
        for k in self.SUM_KEYS:
          sums[k] = s[k] if k not in sums else sums[k] + s[k]
        for k in self.EP_KEYS:
          per_ep[k].append(s[k])
        if layout is None:
          layout = s['pos_layout']
        if (i + 1) % 50 == 0:
          print(f'[explore]   {i + 1}/{len(self.episodes)}')
    out = {k: np.asarray(v) for k, v in sums.items()}
    out.update({f'ep_{k}': np.asarray(v) for k, v in per_ep.items()})
    out['pos_layout'] = layout
    return out

  def __getitem__(self, key):
    return self.d[key]


# ---------------------------------------------------------------- explorer

class DataExplorer:

  def __init__(self, data_dir: Path, logdir: Path | None):
    self.data_dir = data_dir.expanduser().resolve()
    self.episodes = sorted(self.data_dir.glob('*.npz'))
    if not self.episodes:
      raise FileNotFoundError(f'No .npz episodes in {self.data_dir}')
    self.names = [p.name for p in self.episodes]
    self.source = self._source_names()
    self.val, self.split_note = self._val_split(logdir)
    self.stats = DatasetStats(self.episodes)
    self.success = self.stats['ep_final_reward'] > 0
    self.lengths = self.stats['ep_length']
    self._density = None

  def _source_names(self) -> dict[str, str]:
    manifest = self.data_dir / 'manifest.json'
    if not manifest.exists():
      return {}
    written = json.loads(manifest.read_text()).get('written', {})
    return {chunk: src.removeprefix('episode_').removesuffix('.npz')
            for src, chunk in written.items()}

  def _val_split(self, logdir):
    if logdir is not None:
      f = Path(logdir).expanduser() / 'val_files.txt'
      if f.exists():
        val = {line.strip() for line in f.read_text().splitlines()
               if line.strip()}
        return val, f'from {f}'
    return (split_demos(self.names),
            'recomputed with val_frac=0.1, seed=0, the split a new training '
            'run makes. No val_files.txt in the logdir, so that checkpoint may '
            'have been trained on every episode (check val_frac in its '
            'config.yaml)')

  # -------------------------------------------------------------- helpers

  def label(self, i: int) -> str:
    name = self.names[i]
    src = self.source.get(name, name[:24])
    outcome = 'success' if self.success[i] else 'fail'
    split = 'val' if name in self.val else 'train'
    return f'{src} | {outcome} | {split} | {self.lengths[i]} steps'

  def choices(self) -> list[tuple[str, str]]:
    order = sorted(range(len(self.names)), key=lambda i: self._sort_key(i))
    return [(self.label(i), self.names[i]) for i in order]

  def _sort_key(self, i):
    src = self.source.get(self.names[i], self.names[i])
    parts = src.split('_')
    try:
      return (parts[0], int(parts[1]), int(parts[2]))
    except (IndexError, ValueError):
      return (src, 0, 0)

  def index(self, name: str) -> int:
    return self.names.index(name)

  @lru_cache(maxsize=4)
  def episode(self, name: str):
    ep = load_episode(self.data_dir / name)
    ep['mag'] = np.linalg.norm(ep['forces_local'], axis=-1)
    return ep

  # -------------------------------------------------------------- overview

  def overview_text(self) -> str:
    n = len(self.names)
    ns = int(self.success.sum())
    val = np.array([x in self.val for x in self.names])
    total = int(self.lengths.sum())
    groups = {}
    for i, name in enumerate(self.names):
      g = '_'.join(self.source.get(name, 'unknown_x').split('_')[:2])
      groups.setdefault(g, [0, 0])
      groups[g][0 if self.success[i] else 1] += 1
    nonfinite = int(self.stats['ep_nonfinite'].sum())
    nz = self.stats['ep_nonzero_rewards']
    lines = [
        f'### {n} episodes, {total:,} timesteps '
        f'({total / FREQ / 60:.1f} minutes of robot time at {FREQ:g} Hz)',
        '',
        '| | episodes | success | fail |',
        '|---|---|---|---|',
        f'| all | {n} | {ns} ({100 * ns / n:.0f}%) | {n - ns} |',
        f'| train | {int((~val).sum())} | {int((self.success & ~val).sum())} '
        f'| {int((~self.success & ~val).sum())} |',
        f'| val (held out) | {int(val.sum())} | '
        f'{int((self.success & val).sum())} | '
        f'{int((~self.success & val).sum())} |',
    ]
    for g, (s, f) in sorted(groups.items()):
      lines.append(f'| source `{g}_*` | {s + f} | {s} | {f} |')
    lines += [
        '',
        f'- Episode length: min {self.lengths.min()}, median '
        f'{int(np.median(self.lengths))}, max {self.lengths.max()} steps '
        f'({self.lengths.min() / FREQ:.1f}-{self.lengths.max() / FREQ:.1f} s).',
        f'- Reward: {int((nz == 1).sum())} episodes have exactly one non-zero '
        f'reward (the last step: +1 success, -1 fail); '
        f'{int((nz != 1).sum())} have something else.',
        f'- Successful episodes end with `is_terminal=True` '
        f'({int(self.stats["ep_terminal"][self.success].sum())}/{ns}); failed '
        f'ones are cut off (`is_terminal=False`, '
        f'{int((~self.stats["ep_terminal"][~self.success]).sum())}/{n - ns}).',
        f'- Non-finite values (NaN/inf) in the data: {nonfinite}.',
        f'- Validation split: {self.split_note}.',
        f'- A training sample is a window of 65 consecutive steps '
        f'(0.65 s). The dataset holds about {total // 65:,} non-overlapping '
        'windows.',
    ]
    return '\n'.join(lines)

  def overview_figure(self):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), constrained_layout=True)
    bins = np.linspace(self.lengths.min(), self.lengths.max(), 40)
    ax = axes[0]
    ax.hist(self.lengths[self.success], bins, alpha=0.7, label='success',
            color='tab:green')
    ax.hist(self.lengths[~self.success], bins, alpha=0.7, label='fail',
            color='tab:red')
    ax.set_xlabel('episode length (steps, 100 per second)')
    ax.set_ylabel('# episodes')
    ax.set_title('How long are the episodes?')
    ax.legend()

    ax = axes[1]
    mf = self.stats['ep_max_force']
    ax.scatter(self.lengths[self.success], mf[self.success], s=8,
               c='tab:green', label='success')
    ax.scatter(self.lengths[~self.success], mf[~self.success], s=8,
               c='tab:red', label='fail')
    val = np.array([x in self.val for x in self.names])
    ax.scatter(self.lengths[val], mf[val], s=40, facecolors='none',
               edgecolors='black', label='val (held out)')
    ax.set_xlabel('episode length (steps)')
    ax.set_ylabel('max taxel |force| in episode (counts)')
    ax.set_title('One dot per episode')
    ax.legend(fontsize=8)

    ax = axes[2]
    mean_f = self.stats['ep_mean_force_finger']
    pos = np.arange(len(FINGERS))
    for k, (mask, c, lab) in enumerate((
        (self.success, 'tab:green', 'success'),
        (~self.success, 'tab:red', 'fail'))):
      parts = ax.violinplot([mean_f[mask, i] for i in pos],
                            pos + (k - 0.5) * 0.35, widths=0.33,
                            showmedians=True)
      for b in parts['bodies']:
        b.set_facecolor(c)
        b.set_alpha(0.5)
      for key in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        parts[key].set_color(c)
      ax.plot([], [], color=c, label=lab)
    ax.set_xticks(pos, FINGERS)
    ax.set_ylabel('mean taxel |force| over episode')
    ax.set_title('Average contact per finger, per episode')
    ax.legend(fontsize=8)
    img = figure_to_image(fig)
    plt.close(fig)
    return img

  def episode_table(self):
    rows = []
    for i in sorted(range(len(self.names)), key=self._sort_key):
      name = self.names[i]
      mf = self.stats['ep_mean_force_finger'][i]
      rows.append([
          self.source.get(name, ''), 'success' if self.success[i] else 'fail',
          'val' if name in self.val else 'train', int(self.lengths[i]),
          round(float(self.stats['ep_max_force'][i]), 1),
          *[round(float(x), 1) for x in mf], name])
    headers = ['source', 'outcome', 'split', 'steps', 'max |F|',
               *[f'mean |F| {f}' for f in FINGERS], 'chunk file']
    return headers, rows

  # -------------------------------------------------------------- episode

  def render_episode(self, name, step, color_mode, show_window):
    ep = self.episode(name)
    t_len = len(ep['action'])
    step = int(np.clip(step, 0, t_len - 1))
    mag = ep['mag']
    pos = ep['positions'][step]
    vals = mag[step]
    if color_mode.startswith('Fixed'):
      vmax = float(np.percentile(mag, 99.5))
    else:
      vmax = max(float(vals.max()), 1e-6)

    # Hand at the current step: two projections + per-patch bars.
    fig = plt.figure(figsize=(16, 5.2), constrained_layout=True)
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.5])
    order = np.argsort(vals)  # draw strong contacts on top
    for k, (dims, title, labels) in enumerate((
        ((0, 2), 'Front view (x, z)', ('x (m)', 'z (m)')),
        ((1, 2), 'Side view (y, z)', ('y (m)', 'z (m)')),
        ((0, 1), 'Top view (x, y) - as in model visualizer', ('x (m)', 'y (m)')))):
      ax = fig.add_subplot(grid[0, k])
      sc = ax.scatter(pos[order, dims[0]], pos[order, dims[1]], c=vals[order],
                      s=14, cmap='magma', vmin=0, vmax=vmax,
                      edgecolors='0.6', linewidths=0.2)
      if k == 0:
        for f in FINGERS:
          m = TAXEL_FINGER == FINGERS.index(f)
          ax.text(pos[m, 0].mean(), pos[m, 2].min() - 0.008, f, fontsize=8,
                  color=FINGER_COLORS[f], ha='center', va='top',
                  weight='bold')
      ax.set_title(title, fontsize=10)
      ax.set_xlabel(labels[0])
      ax.set_ylabel(labels[1])
      ax.set_aspect('equal', adjustable='datalim')
      ax.set_facecolor('0.85')
    fig.colorbar(sc, ax=fig.axes, shrink=0.8, label='|force| (Xela counts)',
                 location='left')
    ax = fig.add_subplot(grid[0, 3])
    pc = patch_curves(mag[step:step + 1])[0]
    ax.barh(np.arange(len(PATCHES)), pc,
            color=[FINGER_COLORS[f] for _, f, _ in PATCHES])
    ax.set_yticks(np.arange(len(PATCHES)), [p for p, _, _ in PATCHES],
                  fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(float(np.percentile(patch_curves(mag), 99.5)), 1))
    ax.set_xlabel('mean |force| over patch taxels')
    ax.set_title('Contact per sensor patch', fontsize=10)
    fig.suptitle(f'{self.label(self.index(name))}    step {step} '
                 f'(t = {step / FREQ:.2f} s)', fontsize=11)
    hand = figure_to_image(fig)
    plt.close(fig)

    # Timeline of the whole episode.
    t = np.arange(t_len)
    fig, axes = plt.subplots(
        4, 1, figsize=(16, 11), sharex=True, constrained_layout=True,
        gridspec_kw={'height_ratios': [2.2, 1.4, 2.2, 0.8]})
    pcs = patch_curves(mag)
    im = axes[0].imshow(
        pcs.T, aspect='auto', cmap='magma', interpolation='nearest',
        extent=[-0.5, t_len - 0.5, len(PATCHES) - 0.5, -0.5],
        vmin=0, vmax=max(float(np.percentile(pcs, 99.5)), 1))
    axes[0].set_yticks(np.arange(len(PATCHES)), [p for p, _, _ in PATCHES],
                       fontsize=7)
    axes[0].set_title('Contact timeline: mean |force| per sensor patch '
                      '(bright = pressed)', fontsize=10)
    fig.colorbar(im, ax=axes[0], pad=0.01, label='counts')

    fc = finger_curves(mag)
    for i, f in enumerate(FINGERS):
      axes[1].plot(t, fc[:, i], color=FINGER_COLORS[f], label=f, lw=1)
    axes[1].set_ylabel('mean |force|')
    axes[1].set_title('Contact per finger', fontsize=10)
    axes[1].legend(fontsize=8, ncol=5, loc='upper left')

    act = ep['action']
    styles = ('-', '--', '-.', ':')
    for j, jn in enumerate(JOINTS):
      axes[2].plot(t, act[:, j], color=FINGER_COLORS[JOINT_FINGER[j]],
                   ls=styles[j % 4], lw=1, label=jn)
    axes[2].set_ylim(-1.05, 1.05)
    axes[2].set_ylabel('action (scaled joint target)')
    axes[2].set_title('Actions: 16 joint targets, colour = finger, '
                      'line style = joint', fontsize=10)
    axes[2].legend(fontsize=7, ncol=8, loc='lower left')

    rew = ep['reward']
    axes[3].plot(t, rew, color='black', lw=1, label='reward')
    for key, c, y in (('is_first', 'tab:blue', 0.3),
                      ('is_last', 'tab:orange', 0.6),
                      ('is_terminal', 'tab:red', 0.9)):
      idx = np.flatnonzero(ep[key])
      axes[3].scatter(idx, np.zeros_like(idx) + y, marker='v', s=60,
                      color=c, label=key, zorder=3)
    axes[3].set_ylim(-1.2, 1.2)
    axes[3].set_ylabel('reward')
    axes[3].set_xlabel(f'step ({FREQ:g} steps = 1 s)')
    axes[3].legend(fontsize=8, ncol=4, loc='lower left')

    for ax in axes:
      ax.axvline(step, color='cyan' if ax is axes[0] else 'black',
                 ls=':', lw=1.5)
      if show_window:
        start = min(step, max(t_len - WINDOW, 0))
        ax.axvspan(start - 0.5, start + WINDOW - 0.5, color='tab:cyan',
                   alpha=0.15 if ax is not axes[0] else 0.0)
        if ax is axes[0]:
          ax.axvline(start - 0.5, color='cyan', lw=1)
          ax.axvline(start + WINDOW - 0.5, color='cyan', lw=1)
    timeline = figure_to_image(fig)
    plt.close(fig)

    q = (act[step] + 1) / 2 * (ACTION_HIGH - ACTION_LOW) + ACTION_LOW
    lines = [
        f'{self.label(self.index(name))}',
        f'chunk file: {name}',
        f'step {step} / {t_len - 1}   (t = {step / FREQ:.2f} s)',
        '',
        f'reward: {rew[step]:+.0f}   is_first={bool(ep["is_first"][step])}  '
        f'is_last={bool(ep["is_last"][step])}  '
        f'is_terminal={bool(ep["is_terminal"][step])}',
        f'strongest taxel: #{int(vals.argmax())} '
        f'({PATCHES[TAXEL_PATCH[vals.argmax()]][0]}) '
        f'|F| = {vals.max():.0f} counts',
        f'taxels above 100 counts: {int((vals > 100).sum())} / {N_TAXELS}',
        '',
        'action[t] (scaled -> joint target in rad):',
    ]
    for j in range(0, 16, 4):
      lines.append('  ' + '   '.join(
          f'{JOINTS[k]:>6} {act[step, k]:+.3f} -> {q[k]:+.2f}'
          for k in range(j, j + 4)))
    if show_window:
      start = min(step, max(t_len - WINDOW, 0))
      lines += ['', f'Cyan band: a {WINDOW}-step window starting at {start} '
                f'(what the model visualizer feeds the model with '
                f'"Window start" = {start}).']
    return hand, timeline, '\n'.join(lines)

  # -------------------------------------------------------------- tactile

  def tactile_figure(self):
    s = self.stats
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    axn = ('x', 'y', 'z')
    colors = ('tab:red', 'tab:green', 'tab:blue')

    def hist_line(ax, h, bins, label, color):
      c = 0.5 * (bins[1:] + bins[:-1])
      ax.plot(c, h / h.sum() / np.diff(bins), color=color, label=label)

    for k in range(3):
      hist_line(axes[0, 0], s['h_fraw'][k], B_FRAW, axn[k], colors[k])
      hist_line(axes[0, 1], s['h_fz'][k], B_FZ, axn[k], colors[k])
      hist_line(axes[0, 2], s['h_fsym'][k], B_FSYM, axn[k], colors[k])
      hist_line(axes[1, 0], s['h_pos'][k], B_POS, axn[k], colors[k])
    axes[0, 0].set_title('1. Raw forces_local (what is on disk)', fontsize=10)
    axes[0, 0].set_xlabel('Xela counts')
    axes[0, 1].set_title('2. After z-score  (x - mean) / std  (forces_norm)',
                         fontsize=10)
    axes[0, 1].set_xlabel('standard deviations')
    axes[0, 2].set_title('3. After symlog: what the encoder sees', fontsize=10)
    axes[0, 2].set_xlabel('symlog(z)')
    axes[0, 2].axvspan(-np.log(2), np.log(2), color='gold', alpha=0.25,
                       label='|z| < 1: symlog ~ linear')
    for ax in axes[0]:
      ax.set_yscale('log')
      ax.set_ylabel('density (log scale)')
      ax.legend(fontsize=8)
    axes[1, 0].set_title('positions per axis (metres, NOT normalized)',
                         fontsize=10)
    axes[1, 0].set_xlabel('m')
    axes[1, 0].set_ylabel('density')
    axes[1, 0].legend(fontsize=8)

    # Loss balance: per-dim temporal variance (what the model must predict).
    ax = axes[1, 1]
    pv = s['ep_p_tvar']
    fv = s['ep_fz_tvar']
    ax.hist(np.log10(fv + 1e-12), 40, alpha=0.7, label='forces (z + symlog)')
    ax.hist(np.log10(pv + 1e-12), 40, alpha=0.7, label='positions (raw m)')
    ax.set_xlabel('log10 of mean per-dimension variance over time')
    ax.set_ylabel('# episodes')
    ax.set_title(f'How much does each signal move?  forces / positions '
                 f'~ {np.median(fv) / np.median(pv):,.0f}x', fontsize=10)
    ax.legend(fontsize=8)

    # Per-taxel extreme values.
    ax = axes[1, 2]
    h = s['h_mag_taxel']
    edges = B_MAG
    frac_above = {thr: h[:, np.searchsorted(edges, thr):].sum(1) / h.sum(1)
                  for thr in (100, 300, 1000)}
    for thr, v in frac_above.items():
      ax.plot(np.sort(v)[::-1] * 100, label=f'|F| > {thr}')
    ax.set_yscale('symlog', linthresh=0.1)
    ax.set_xlabel('taxels, sorted from most to least touched')
    ax.set_ylabel('% of all timesteps')
    ax.set_title('How often is each taxel pressed?', fontsize=10)
    ax.legend(fontsize=8)
    img = figure_to_image(fig)
    plt.close(fig)

    f_mean = s['f_sum'] / (self.lengths.sum() * N_TAXELS)
    f_std = np.sqrt(s['f_sq'] / (self.lengths.sum() * N_TAXELS) - f_mean ** 2)
    p_mean = s['p_sum'] / (self.lengths.sum() * N_TAXELS)
    p_std = np.sqrt(s['p_sq'] / (self.lengths.sum() * N_TAXELS) - p_mean ** 2)
    fz_lin = s['h_fsym'][:, np.abs(0.5 * (B_FSYM[1:] + B_FSYM[:-1]))
                         < np.log(2)].sum() / s['h_fsym'].sum()
    zeros = s['zero_taxel']
    text = (
        'Whole-dataset statistics (every timestep, every taxel):\n'
        f'  forces mean x/y/z: {f_mean.round(2)}   std: {f_std.round(2)}\n'
        f'  config forces_norm  mean: {FORCE_MEAN}   std: {FORCE_STD}\n'
        f'  positions mean x/y/z (m): {p_mean.round(4)}   std: '
        f'{p_std.round(4)}\n'
        f'  share of z-scored force values with |z| < 1 (symlog linear): '
        f'{100 * fz_lin:.1f}%\n'
        f'  taxel readings exactly 0 (outlier removal): '
        f'{int(zeros.sum()):,} of {int(self.lengths.sum()) * N_TAXELS:,} '
        f'({100 * zeros.sum() / (self.lengths.sum() * N_TAXELS):.3f}%)\n'
        f'  median position motion per step: '
        f'{1000 * np.median(s["ep_pos_motion"]):.3f} mm\n')
    return img, text

  def taxel_map(self, threshold):
    s = self.stats
    h = s['h_mag_taxel']
    frac = h[:, np.searchsorted(B_MAG, threshold):].sum(1) / h.sum(1)
    cdf = np.cumsum(h, 1) / h.sum(1, keepdims=True)
    median = B_MAG[1:][np.array([np.searchsorted(c, 0.5) for c in cdf])]
    pos = s['pos_layout']
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2), constrained_layout=True)
    sc = axes[0].scatter(pos[:, 0], pos[:, 2], c=100 * frac, s=22,
                         cmap='viridis', edgecolors='0.5', linewidths=0.2,
                         vmin=0, vmax=100)
    fig.colorbar(sc, ax=axes[0], label='% of timesteps')
    axes[0].set_title(f'Taxel activity: % of time |force| > {threshold:g}',
                      fontsize=10)
    sc = axes[1].scatter(pos[:, 0], pos[:, 2], c=median, s=22,
                         cmap='magma', edgecolors='0.5', linewidths=0.2,
                         norm=LogNorm(vmin=1, vmax=max(median.max(), 10)))
    fig.colorbar(sc, ax=axes[1], label='median |force| (counts, log)')
    axes[1].set_title('Typical (median) |force| of each taxel', fontsize=10)
    for ax in axes[:2]:
      ax.set_facecolor('0.85')
      ax.set_xlabel('x (m)')
      ax.set_ylabel('z (m)')
      ax.set_aspect('equal', adjustable='datalim')
      for f in FINGERS:
        m = TAXEL_FINGER == FINGERS.index(f)
        ax.text(pos[m, 0].mean(), pos[m, 2].min() - 0.008, f, fontsize=8,
                color=FINGER_COLORS[f], weight='bold', ha='center',
                va='top')
    pf = np.array([frac[s_].mean() for s_ in PATCH_SLICES])
    axes[2].barh(np.arange(len(PATCHES)), 100 * pf,
                 color=[FINGER_COLORS[f] for _, f, _ in PATCHES])
    axes[2].set_yticks(np.arange(len(PATCHES)), [p for p, _, _ in PATCHES],
                       fontsize=8)
    axes[2].invert_yaxis()
    axes[2].set_xlim(0, 100)
    axes[2].set_xlabel(f'% of time |force| > {threshold:g} (patch average)')
    axes[2].set_title('Activity per patch', fontsize=10)
    img = figure_to_image(fig)
    plt.close(fig)
    return img

  # -------------------------------------------------------------- actions

  def actions_figure(self):
    s = self.stats
    n = self.lengths.sum()
    fig = plt.figure(figsize=(16, 11), constrained_layout=True)
    grid = fig.add_gridspec(3, 6)
    c = 0.5 * (B_ACT[1:] + B_ACT[:-1])
    for j in range(16):
      row, col = (j // 6, j % 6) if j < 12 else (2, j - 12)
      ax = fig.add_subplot(grid[row, col])
      ax.fill_between(c, s['h_act'][j] / n, step='mid',
                      color=FINGER_COLORS[JOINT_FINGER[j]], alpha=0.7)
      ax.set_title(JOINTS[j], fontsize=9)
      ax.set_xlim(-1, 1)
      ax.set_yticks([])
      ax.axvline(-0.25, color='black', lw=0.5, ls=':')   # 0 rad = open pose
    mean = s['a_sum'] / n
    cov = s['a_outer'] / n - np.outer(mean, mean)
    sd = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    corr = cov / np.outer(sd, sd)
    ax = fig.add_subplot(grid[2, 4:])
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(16), JOINTS, rotation=90, fontsize=7)
    ax.set_yticks(range(16), JOINTS, fontsize=7)
    ax.set_title('Joint correlation (+1 move together, -1 opposite)',
                 fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle('Distribution of each joint target over all timesteps '
                 '(x: scaled action in [-1, 1]; dotted line = 0 rad)',
                 fontsize=11)
    img = figure_to_image(fig)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(16, 3.6), constrained_layout=True)
    hd = s['h_dact']
    still = hd[:, 0] / hd.sum(1)
    cdf = np.cumsum(hd, 1) / hd.sum(1, keepdims=True)
    med = np.array([B_DACT[1:][np.searchsorted(cdf[j], 0.99)]
                    for j in range(16)])
    x = np.arange(16)
    ax.bar(x - 0.2, 100 * still, 0.4, label='% of steps with no change',
           color=[FINGER_COLORS[f] for f in JOINT_FINGER])
    ax2 = ax.twinx()
    ax2.plot(x + 0.2, med, 'ko', label='99th percentile |a[t+1] - a[t]|')
    ax2.set_yscale('log')
    ax2.set_ylabel('|change per step| (99th pct)')
    ax.set_xticks(x, JOINTS, fontsize=8)
    ax.set_ylabel('% steps unchanged')
    ax.set_title('How smoothly do the actions change? (commands are held '
                 'until the next one arrives)', fontsize=10)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    img2 = figure_to_image(fig)
    plt.close(fig)
    return img, img2

  # -------------------------------------------------------------- outcome

  def outcome_figure(self, align):
    s = self.stats
    key = 'ep_end_finger' if align.startswith('End') else 'ep_start_finger'
    curves = s[key]  # (E, ALIGN, 5)
    if key == 'ep_end_finger':
      t = (np.arange(ALIGN) - ALIGN + 1) / FREQ
      xlabel = 'seconds before the last step'
    else:
      t = np.arange(ALIGN) / FREQ
      xlabel = 'seconds after the first step'
    fig, axes = plt.subplots(1, len(FINGERS), figsize=(16, 3.8),
                             sharey=True, constrained_layout=True)
    for i, f in enumerate(FINGERS):
      ax = axes[i]
      for mask, c, lab in ((self.success, 'tab:green', 'success'),
                           (~self.success, 'tab:red', 'fail')):
        x = curves[mask, :, i]
        med = np.nanmedian(x, 0)
        lo, hi = np.nanpercentile(x, 25, 0), np.nanpercentile(x, 75, 0)
        ax.plot(t, med, color=c, label=f'{lab} median (n={mask.sum()})')
        ax.fill_between(t, lo, hi, color=c, alpha=0.2)
      ax.set_title(f, color=FINGER_COLORS[f])
      ax.set_xlabel(xlabel)
    axes[0].set_ylabel('mean taxel |force| (counts)')
    axes[0].legend(fontsize=7)
    fig.suptitle('Success vs fail: per-finger contact, line = median over '
                 'episodes, band = middle 50% of episodes', fontsize=11)
    img = figure_to_image(fig)
    plt.close(fig)
    return img

  # -------------------------------------------------------------- coverage

  def density(self):
    if self._density is None:
      from data_density import DataDensity
      self._density = DataDensity(self.episodes)
    return self._density

  def coverage_figure(self, space, color_by, highlight):
    d = self.density()
    emb = d.emb[space]
    xy = emb[:, :2]
    ep_idx = d.ep_index
    lo = np.percentile(xy, 0.1, 0)
    hi = np.percentile(xy, 99.9, 0)
    pad = 0.05 * (hi - lo)
    rng = [[lo[0] - pad[0], hi[0] + pad[0]], [lo[1] - pad[1], hi[1] + pad[1]]]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    def hist(ax, mask, cmap, title):
      h, xe, ye = np.histogram2d(xy[mask, 0], xy[mask, 1], 200, range=rng)
      im = ax.imshow(np.where(h.T > 0, h.T, np.nan), origin='lower',
                     extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect='auto',
                     cmap=cmap, norm=LogNorm(vmin=1, vmax=max(h.max(), 2)))
      ax.set_title(title, fontsize=10)
      ax.set_xlabel('PC1')
      return im

    im = hist(axes[0], np.ones(len(xy), bool), 'viridis',
              f'All {len(xy):,} timesteps, {space} PC1-PC2')
    fig.colorbar(im, ax=axes[0], label='steps per bin (log)')
    axes[0].set_ylabel('PC2')

    if color_by == 'Outcome':
      groups = ((self.success, 'Greens', 'success episodes'),
                (~self.success, 'Reds', 'fail episodes'))
    else:
      val = np.array([x in self.val for x in self.names])
      groups = ((~val, 'Blues', 'train episodes'),
                (val, 'Oranges', 'val (held-out) episodes'))
    for ax, (ep_mask, cmap, title) in zip(axes[1:], groups):
      hist(ax, ep_mask[ep_idx], cmap, title)
      ax.set_xlim(rng[0])
      ax.set_ylim(rng[1])

    if highlight:
      i = self.index(highlight)
      pts = d.episode_points(space, highlight)
      for ax in axes:
        ax.plot(pts[:, 0], pts[:, 1], color='black', lw=0.8, alpha=0.8)
        ax.plot(pts[0, 0], pts[0, 1], 'o', color='white', mec='black', ms=7)
        ax.plot(pts[-1, 0], pts[-1, 1], 's', color='white', mec='black',
                ms=7)
      fig.suptitle(f'Black line: {self.label(i)} (circle = start, square = '
                   'end)', fontsize=10)
    img = figure_to_image(fig)
    plt.close(fig)
    text = (f'{space}: the first {d.emb[space].shape[1]} principal components '
            f'keep {100 * d.explained[space]:.1f}% of the variance. '
            'Only PC1 and PC2 are drawn.')
    return img, text


# ---------------------------------------------------------------- app

def build_app(ex: DataExplorer) -> gr.Blocks:
  choices = ex.choices()
  default = choices[0][1]

  with gr.Blocks(title='Mechanical Pen Data Explorer') as demo:
    gr.Markdown(
        '# Mechanical pen data explorer\n'
        f'`{ex.data_dir}`, the demos the world model trains on. '
        'The course in `eval/mechanical_pen/course/` explains each tab.')

    with gr.Tab('1. Overview'):
      gr.Markdown(ex.overview_text())
      gr.Image(ex.overview_figure(), label='Episodes at a glance')
      headers, rows = ex.episode_table()
      gr.Dataframe(rows, headers=headers, label='All episodes (click a '
                   'column header to sort)', interactive=False,
                   max_height=420)

    with gr.Tab('2. Episode'):
      with gr.Row():
        episode = gr.Dropdown(choices, value=default, label='Episode',
                              scale=3)
        color_mode = gr.Radio(
            ['Fixed per episode', 'Rescale every step'],
            value='Fixed per episode', label='Colour scale', scale=1)
        show_window = gr.Checkbox(True, label='Show 64-step model window',
                                  scale=1)
      step = gr.Slider(0, int(ex.lengths[ex.index(default)]) - 1, value=0,
                       step=1, label='Step')
      with gr.Row():
        back = gr.Button('<< -100')
        back1 = gr.Button('< -10')
        fwd1 = gr.Button('+10 >')
        fwd = gr.Button('+100 >>')
        peak = gr.Button('Jump to strongest contact')
        last = gr.Button('Jump to last step')
      hand = gr.Image(label='Hand at this step')
      ep_text = gr.Textbox(label='Values at this step', lines=14)
      timeline = gr.Image(label='Whole episode')

      ep_inputs = [episode, step, color_mode, show_window]
      ep_outputs = [hand, timeline, ep_text]

      def on_episode(name):
        n = int(ex.lengths[ex.index(name)])
        return gr.update(maximum=n - 1, value=0)

      episode.change(on_episode, episode, step)
      for ctrl in ep_inputs:
        ctrl.change(ex.render_episode, ep_inputs, ep_outputs)
      demo.load(ex.render_episode, ep_inputs, ep_outputs)

      def shift(delta):
        def fn(name, st):
          n = int(ex.lengths[ex.index(name)])
          return int(np.clip(st + delta, 0, n - 1))
        return fn

      for btn, delta in ((back, -100), (back1, -10), (fwd1, 10), (fwd, 100)):
        btn.click(shift(delta), [episode, step], step)
      peak.click(lambda name: int(ex.episode(name)['mag'].max(1).argmax()),
                 episode, step)
      last.click(lambda name: int(ex.lengths[ex.index(name)]) - 1,
                 episode, step)

    with gr.Tab('3. Tactile stats'):
      tact_img, tact_text = ex.tactile_figure()
      gr.Image(tact_img, label='Value distributions')
      gr.Textbox(tact_text, label='Numbers', lines=8)
      threshold = gr.Slider(10, 2000, value=100, step=10,
                            label='Contact threshold |force| (counts)')
      taxel_img = gr.Image(label='Which taxels get touched?')
      threshold.change(ex.taxel_map, threshold, taxel_img)
      demo.load(ex.taxel_map, threshold, taxel_img)

    with gr.Tab('4. Actions'):
      a1, a2 = ex.actions_figure()
      gr.Image(a1, label='Joint targets')
      gr.Image(a2, label='Action smoothness')

    with gr.Tab('5. Success vs fail'):
      align = gr.Radio(['End of episode', 'Start of episode'],
                       value='End of episode', label='Align episodes at')
      outcome_img = gr.Image(label='Contact over time')
      align.change(ex.outcome_figure, align, outcome_img)
      demo.load(ex.outcome_figure, align, outcome_img)

    with gr.Tab('6. Coverage map'):
      gr.Markdown(
          'Every timestep squeezed to 2 numbers with PCA (same embedding as '
          'the density view in the model visualizer). Press **Draw** - '
          'the first time it can take a few minutes if the cache '
          '(`.cache/density_*.npz`) has not been built yet.')
      with gr.Row():
        space = gr.Radio(['Observation', 'Forces', 'Positions', 'Action'],
                         value='Observation', label='Feature space')
        color_by = gr.Radio(['Outcome', 'Train / val split'],
                            value='Outcome', label='Split the map by')
        highlight = gr.Dropdown([('(none)', '')] + choices, value='',
                                label='Draw one episode on top')
      draw = gr.Button('Draw', variant='primary')
      cov_img = gr.Image(label='Coverage')
      cov_text = gr.Markdown()
      draw.click(ex.coverage_figure, [space, color_by, highlight],
                 [cov_img, cov_text])

  return demo


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--data', type=Path,
                      default=_REPO_ROOT / 'data/leap/mechanical_pen')
  parser.add_argument('--logdir', type=Path,
                      default=_REPO_ROOT / 'logdir/model/leap_mechanical_pen/0/0',
                      help='Optional: reads <logdir>/val_files.txt for the split')
  parser.add_argument('--host', default='0.0.0.0')
  parser.add_argument('--port', type=int, default=7861)
  parser.add_argument('--share', action='store_true')
  args = parser.parse_args()

  ex = DataExplorer(args.data, args.logdir)
  demo = build_app(ex)
  try:
    demo.launch(server_name=args.host, server_port=args.port,
                share=args.share)
  except OSError as exc:
    if 'Cannot find empty port' not in str(exc):
      raise
    print(f'Port {args.port} busy; picking a free port...')
    demo.launch(server_name=args.host, server_port=None, share=args.share)


if __name__ == '__main__':
  main()

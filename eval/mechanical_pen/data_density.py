"""Dataset density / coverage view for the world model visualizer.

Every timestep of every episode in the data dir is projected into a small PCA
space (one per feature space). The visualizer then shows where the selected
step sits inside the cloud of all training data and counts how many dataset
points lie within a radius around it, so you can judge whether the model has
seen similar states before.

Feature spaces (per timestep):
  Forces       symlog(per-axis z-score of forces_local), 368*3 dims
  Positions    per-axis z-score of positions, 368*3 dims
  Action       raw action (already in [-1, 1]), 16 dims
  Observation  Forces and Positions PCs, each block scaled to unit variance

The embedding is computed in streaming passes over the .npz files (the full
dataset does not fit in memory as raw arrays) and cached under
eval/mechanical_pen/.cache/ keyed by the episode file list.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from sklearn.neighbors import KDTree

NUM_DIMS = 8  # PCA dims used for neighbour search (first two are plotted)
HIST_BINS = 300
SPACES = ('Observation', 'Forces', 'Positions', 'Action')
_CACHE_DIR = Path(__file__).resolve().parent / '.cache'
_CACHE_VERSION = 1


def _symlog(x):
  return np.sign(x) * np.log1p(np.abs(x))


def _load(path: Path, keys=('forces_local', 'positions', 'action')):
  raw = np.load(path)
  return {k: np.asarray(raw[k], np.float32) for k in keys}


def _map_files(fn, files, workers=16):
  # zlib decompression releases the GIL, so threads give a real speedup.
  with ThreadPoolExecutor(workers) as pool:
    yield from pool.map(fn, files)


class DataDensity:

  def __init__(self, episodes: list[Path]):
    self.episodes = list(episodes)
    self.name_to_index = {p.name: i for i, p in enumerate(self.episodes)}
    self._trees: dict[str, KDTree] = {}
    self._hists: dict[str, tuple] = {}
    self._load_or_build()

  # ------------------------------------------------------------------ build

  def _cache_path(self) -> Path:
    h = hashlib.sha1()
    h.update(str(_CACHE_VERSION).encode())
    for p in self.episodes:
      h.update(p.name.encode())
    return _CACHE_DIR / f'density_{h.hexdigest()[:16]}.npz'

  def _load_or_build(self):
    path = self._cache_path()
    if path.exists():
      data = np.load(path)
      self.ep_index = data['ep_index']
      self.step_index = data['step_index']
      self.ep_offsets = data['ep_offsets']
      self.emb = {s: data[f'emb_{s}'] for s in SPACES}
      self.explained = {s: float(data[f'explained_{s}']) for s in SPACES}
      print(f'[density] loaded cached embedding {path}')
      return
    print(f'[density] building PCA embedding of {len(self.episodes)} '
          'episodes (one-off, cached afterwards)...')
    self._build()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, ep_index=self.ep_index, step_index=self.step_index,
        ep_offsets=self.ep_offsets,
        **{f'emb_{s}': self.emb[s] for s in SPACES},
        **{f'explained_{s}': self.explained[s] for s in SPACES})
    print(f'[density] cached to {path}')

  def _build(self):
    files = self.episodes

    # Pass 0: per-axis mean/std of forces and positions on a subsample.
    sub = files[::max(1, len(files) // 40)]
    acc = {k: [np.zeros(3), np.zeros(3), 0] for k in ('forces_local', 'positions')}
    for ep in _map_files(_load, sub):
      for k, a in acc.items():
        x = ep[k].reshape(-1, 3).astype(np.float64)
        a[0] += x.sum(0)
        a[1] += (x ** 2).sum(0)
        a[2] += len(x)
    axis_stats = {}
    for k, (s, s2, n) in acc.items():
      mean = s / n
      std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
      axis_stats[k] = (mean.astype(np.float32), std.astype(np.float32))

    def features(ep):
      fm, fs = axis_stats['forces_local']
      pm, ps = axis_stats['positions']
      t = len(ep['action'])
      return {
          'Forces': _symlog((ep['forces_local'] - fm) / fs).reshape(t, -1),
          'Positions': ((ep['positions'] - pm) / ps).reshape(t, -1),
          'Action': ep['action'].reshape(t, -1),
      }

    def load_features(path):
      return features(_load(path))

    # Pass 1: streaming mean and covariance per base space.
    base = ('Forces', 'Positions', 'Action')
    sums, outer, count = {}, {}, 0
    lengths = []
    for feat in _map_files(load_features, files):
      n = len(feat['Action'])
      lengths.append(n)
      count += n
      for s in base:
        x = feat[s].astype(np.float64)
        if s not in sums:
          sums[s] = np.zeros(x.shape[1])
          outer[s] = np.zeros((x.shape[1], x.shape[1]))
        sums[s] += x.sum(0)
        outer[s] += x.T @ x
    proj = {}
    explained = {}
    for s in base:
      mean = sums[s] / count
      cov = outer[s] / count - np.outer(mean, mean)
      evals, evecs = np.linalg.eigh(cov)
      order = np.argsort(evals)[::-1]
      evals, evecs = np.maximum(evals[order], 0), evecs[:, order]
      k = min(NUM_DIMS, len(evals))
      proj[s] = (mean.astype(np.float32), evecs[:, :k].astype(np.float32))
      explained[s] = float(evals[:k].sum() / max(evals.sum(), 1e-12))

    # Pass 2: project every timestep.
    def project(path):
      feat = load_features(path)
      return {s: (feat[s] - proj[s][0]) @ proj[s][1] for s in base}

    chunks = {s: [] for s in base}
    for out in _map_files(project, files):
      for s in base:
        chunks[s].append(out[s])
    emb = {s: np.concatenate(chunks[s]).astype(np.float32) for s in base}

    # Observation: forces and positions blocks, each scaled to unit total var.
    blocks = []
    for s in ('Forces', 'Positions'):
      blocks.append(emb[s] / np.sqrt(emb[s].var(0).sum() + 1e-12))
    obs = np.concatenate(blocks, 1)
    obs -= obs.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(obs, rowvar=False))
    order = np.argsort(evals)[::-1]
    evals, evecs = evals[order], evecs[:, order]
    emb['Observation'] = (obs @ evecs[:, :NUM_DIMS]).astype(np.float32)
    explained['Observation'] = float(
        evals[:NUM_DIMS].sum() / max(evals.sum(), 1e-12))

    lengths = np.asarray(lengths)
    self.ep_offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    self.ep_index = np.repeat(np.arange(len(files)), lengths).astype(np.int32)
    self.step_index = np.concatenate(
        [np.arange(n) for n in lengths]).astype(np.int32)
    self.emb = emb
    self.explained = explained

  # ------------------------------------------------------------------ query

  def _tree(self, space: str) -> KDTree:
    if space not in self._trees:
      print(f'[density] building KD-tree for {space}...')
      self._trees[space] = KDTree(self.emb[space], leaf_size=64)
    return self._trees[space]

  def spread(self, space: str) -> float:
    """RMS distance of the dataset from its mean; the radius unit."""
    return float(np.sqrt(self.emb[space].var(0).sum()))

  def _hist(self, space: str):
    if space not in self._hists:
      xy = self.emb[space][:, :2]
      lo = np.percentile(xy, 0.1, axis=0)
      hi = np.percentile(xy, 99.9, axis=0)
      pad = 0.05 * (hi - lo)
      lo, hi = lo - pad, hi + pad
      h, xe, ye = np.histogram2d(
          xy[:, 0], xy[:, 1], bins=HIST_BINS,
          range=[[lo[0], hi[0]], [lo[1], hi[1]]])
      self._hists[space] = (h, xe, ye)
    return self._hists[space]

  def episode_points(self, space: str, episode_name: str) -> np.ndarray:
    i = self.name_to_index[episode_name]
    return self.emb[space][self.ep_offsets[i]:self.ep_offsets[i + 1]]

  def window_support(self, space, episode_name, start, length, radius):
    """Neighbour counts (all data, other episodes only) per window step."""
    pts = self.episode_points(space, episode_name)[start:start + length]
    total = self._tree(space).query_radius(pts, radius, count_only=True)
    own_all = self.episode_points(space, episode_name)
    d = np.linalg.norm(pts[:, None] - own_all[None], axis=-1)
    own = (d <= radius).sum(1)
    return total, total - own

  def neighbours(self, space, episode_name, global_step, radius):
    point = self.episode_points(space, episode_name)[global_step]
    idx = self._tree(space).query_radius(point[None], radius)[0]
    return point, idx

  # ------------------------------------------------------------------ plot

  def plot(self, fig, space, episode_name, start, length, step, radius_frac):
    """Draws the density overview, a zoom on the neighbourhood, and the
    neighbour-count curve over the window. Returns a text summary."""
    radius = radius_frac * self.spread(space)
    own_ep = self.name_to_index[episode_name]
    global_step = start + step
    ep_pts = self.episode_points(space, episode_name)
    point, idx = self.neighbours(space, episode_name, global_step, radius)
    nb_eps = self.ep_index[idx]
    other = nb_eps != own_ep
    n_other_eps = len(np.unique(nb_eps[other]))
    total, other_counts = self.window_support(
        space, episode_name, start, length, radius)

    ax_all, ax_zoom, ax_curve = fig.subplots(1, 3)
    h, xe, ye = self._hist(space)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    im = ax_all.imshow(
        np.where(h.T > 0, h.T, np.nan), origin='lower', extent=extent,
        aspect='auto', cmap='viridis', norm=LogNorm(vmin=1, vmax=h.max()))
    fig.colorbar(im, ax=ax_all, label='steps per bin (log)')
    ax_all.plot(ep_pts[:, 0], ep_pts[:, 1], color='white', lw=0.5, alpha=0.6,
                label='this episode')
    win = ep_pts[start:start + length]
    ax_all.plot(win[:, 0], win[:, 1], color='red', lw=1.5, label='window')
    ax_all.plot(point[0], point[1], marker='*', color='red', ms=16,
                mec='white', mew=1.0, label='current step')
    ax_all.set_title(f'All data ({len(self.ep_index):,} steps), '
                     f'{space} PC1-PC2', fontsize=10)
    ax_all.set_xlabel('PC1')
    ax_all.set_ylabel('PC2')
    ax_all.legend(fontsize=7, loc='upper right', facecolor='0.6')

    # Zoom: everything in a box around the current point, neighbours coloured.
    half = max(2.5 * radius, 1e-6)
    xy = self.emb[space][:, :2]
    box = ((np.abs(xy[:, 0] - point[0]) < half)
           & (np.abs(xy[:, 1] - point[1]) < half))
    box_idx = np.flatnonzero(box)
    rng = np.random.default_rng(0)
    if len(box_idx) > 30000:
      box_idx = rng.choice(box_idx, 30000, replace=False)
    ax_zoom.scatter(xy[box_idx, 0], xy[box_idx, 1], s=2, c='lightgray',
                    label='data in view')
    show = idx
    if len(show) > 20000:
      show = rng.choice(show, 20000, replace=False)
    ax_zoom.scatter(xy[show[self.ep_index[show] != own_ep], 0],
                    xy[show[self.ep_index[show] != own_ep], 1],
                    s=3, c='tab:orange',
                    label=f'within r, other eps ({int(other.sum()):,})')
    ax_zoom.scatter(xy[show[self.ep_index[show] == own_ep], 0],
                    xy[show[self.ep_index[show] == own_ep], 1],
                    s=3, c='tab:blue',
                    label=f'within r, same ep ({int((~other).sum()):,})')
    ax_zoom.plot(win[:, 0], win[:, 1], color='red', lw=1)
    ax_zoom.plot(point[0], point[1], marker='*', color='red', ms=16,
                 mec='black', mew=0.8)
    ax_zoom.add_patch(Circle(
        point[:2], radius, fill=False, ls='--', color='black', lw=1))
    ax_zoom.set_xlim(point[0] - half, point[0] + half)
    ax_zoom.set_ylim(point[1] - half, point[1] + half)
    ax_zoom.set_aspect('equal', adjustable='box')
    ax_zoom.set_title(
        f'Neighbourhood, r={radius_frac:.3g} x spread '
        f'({NUM_DIMS}-D distance)', fontsize=10)
    ax_zoom.set_xlabel('PC1')
    ax_zoom.legend(fontsize=7, loc='upper right', markerscale=3)

    t = np.arange(length)
    ax_curve.plot(t, total, color='black', label='all data')
    ax_curve.plot(t, other_counts, color='tab:orange',
                  label='other episodes only')
    ax_curve.set_yscale('symlog', linthresh=1)
    ax_curve.set_ylim(bottom=0)
    ax_curve.axvline(step, color='gray', linestyle=':')
    ax_curve.set_xlabel('Step in window')
    ax_curve.set_ylabel('# dataset steps within r')
    ax_curve.set_title('Data support along window', fontsize=10)
    ax_curve.legend(fontsize=7)

    return (
        f'Data density ({space}, {NUM_DIMS} PCs, '
        f'{100 * self.explained[space]:.1f}% variance):\n'
        f'  radius: {radius_frac:.3g} x spread = {radius:.4g}\n'
        f'  steps within r:         {int(total[step]):,} '
        f'({100 * total[step] / len(self.ep_index):.3f}% of data)\n'
        f'  from other episodes:    {int(other_counts[step]):,} '
        f'across {n_other_eps} / {len(self.episodes) - 1} episodes\n'
        f'  window min/median (other eps): {int(other_counts.min()):,} / '
        f'{int(np.median(other_counts)):,}\n')

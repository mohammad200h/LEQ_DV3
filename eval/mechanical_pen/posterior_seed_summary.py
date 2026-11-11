#!/usr/bin/env python3
"""Combine posterior sweeps run with different seeds: mean and spread for each setting.

  python3 eval/mechanical_pen/posterior_seed_summary.py \
      --sweeps logdir/posterior_sweep_10hz logdir/posterior_sweep_10hz_seed1 \
               logdir/posterior_sweep_10hz_seed2 \
      --configs baseline rep0.01 forces3_wd all --out logdir/posterior_seeds_10hz

Each sweep folder is one seed (run_sweep_posterior.sh with SEED and SWEEP_DIR set).
Writes summary.md and summary.png to --out. Settings missing from a sweep are skipped
for that seed.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from posterior_sweep_summary import summarize  # noqa: E402

# (key, header, format, lower is better)
METRICS = [
    ('post_classes', 'Posterior classes/var, held-out', '{:.2f}', True),
    ('prior_classes', 'Prior classes/var, held-out', '{:.2f}', True),
    ('forces', 'Forces recon, held-out', '{:.2f}', True),
    ('forces_gap', 'Forces held-out / training', '{:.2f}', True),
    ('kl', 'KL held-out (nats)', '{:.2f}', None),
    ('ditto_reward', 'DITTO held-out reward', '{:.3f}', False),
    ('post_cmd_step', 'Policy-on-posterior command change/step', '{:.4f}', True),
    ('post_cmd_err', 'Policy-on-posterior command error (rad)', '{:.3f}', True),
]

GUIDE = """\
## How to read this page

Each setting was trained several times, each time with a different **seed**. The seed
fixes the random choices in training: starting weights, which data is sampled when, and
the random draws of `z`. Changing only the seed shows how much a result moves by chance.

- **Mean:** the average over seeds. Compare settings by this.
- **± (standard deviation):** how far a typical seed lands from the mean. A big ± means
  the result depends on luck.
- **Seeds:** how many seeds finished for that setting.
- A difference between two settings is believable when it is clearly bigger than their
  ± values. If the ranges overlap (mean ± spread of one reaches the other), the
  difference may be chance.
- **Lower is better** for every column except DITTO reward (higher is better) and KL
  (no good direction; it describes how much the posterior and prior disagree).

The column meanings are explained in each sweep's own `summary.md`.
`summary.png` shows each seed as a small dot and the mean as a large dot with a bar
for the ± spread.
"""


def main():
  parser = argparse.ArgumentParser(description='Combine posterior sweeps over seeds.')
  parser.add_argument('--sweeps', type=Path, nargs='+', required=True)
  parser.add_argument('--configs', nargs='+', required=True)
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  # values[name][key] = one value per seed that finished.
  values = {}
  for name in args.configs:
    rows = [summarize(sweep, name) for sweep in args.sweeps if (sweep / name).is_dir()]
    rows = [row for row in rows if row['episodes'] > 0]
    values[name] = {key: [row[key] for row in rows] for key, *_ in METRICS}
    values[name]['seeds'] = len(rows)

  def stats(name, key):
    v = np.array([x for x in values[name][key] if not math.isnan(x)], float)
    if not len(v):
      return math.nan, math.nan
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else math.nan

  lines = ['# Posterior sweep over seeds', '', *GUIDE.splitlines(), '', '## Results', '',
           'Sweeps: ' + ', '.join(f'`{s}`' for s in args.sweeps), '',
           '| Setting | Seeds | ' + ' | '.join(h for _, h, _, _ in METRICS) + ' |',
           '|' + '---|' * (len(METRICS) + 2)]
  for name in args.configs:
    cells = []
    for key, _, spec, _ in METRICS:
      mean, std = stats(name, key)
      if math.isnan(mean):
        cells.append('')
      elif math.isnan(std):
        cells.append(spec.format(mean))
      else:
        cells.append(f'{spec.format(mean)} ± {spec.format(std)}')
    lines.append(f'| {name} | {values[name]["seeds"]} | ' + ' | '.join(cells) + ' |')
  (args.out / 'summary.md').write_text('\n'.join(lines) + '\n')

  fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
  x = np.arange(len(args.configs))
  for ax, (key, header, _, lower) in zip(axes.flat, METRICS):
    hint = {True: ' (lower is better)', False: ' (higher is better)', None: ''}[lower]
    ax.set_title(header + hint, fontsize=10, loc='left')
    for i, name in enumerate(args.configs):
      seeds = [v for v in values[name][key] if not math.isnan(v)]
      ax.plot([i] * len(seeds), seeds, 'o', color='#9fc3ee', ms=6, zorder=2)
      mean, std = stats(name, key)
      if not math.isnan(mean):
        ax.errorbar(i, mean, yerr=0 if math.isnan(std) else std, fmt='o', color='#2a78d6',
                    ms=9, capsize=5, lw=2, markeredgecolor='white', zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(args.configs, rotation=20, ha='right', fontsize=9)
    ax.grid(True, axis='y', color='#e4e3df', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
      ax.spines[side].set_color('#b5b4ae')
    ax.tick_params(colors='#57564f', labelsize=9)
  fig.savefig(args.out / 'summary.png', dpi=90, facecolor='white')
  plt.close(fig)
  print('\n'.join(lines[-(len(args.configs) + 2):]))
  print(f'Wrote {args.out}/summary.md and summary.png')


if __name__ == '__main__':
  main()

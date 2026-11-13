#!/usr/bin/env python3
"""Compare two data sweeps that differ only in dyn.rssm.independent_heads.

Reads, for each rate label and seed, the same per-run numbers as posterior_seed_summary.py
(summarize() in posterior_sweep_summary.py), the rollout metrics.json files, and
ensemble_heads.json from ensemble_heads_eval.py. Writes comparison.md, comparison.csv and
comparison.png to --out.

  python3 eval/mechanical_pen/heads_fix_compare.py \
      --baseline logdir/data_sweep --fixed logdir/data_sweep_indep \
      --labels new10hz new50hz new100hz --seeds 0 1 2 \
      --out eval/mechanical_pen/reports/independent_heads
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from posterior_sweep_summary import summarize  # noqa: E402

# (key, group, header, format, lower is better: True / False / None for no direction)
METRICS = [
    ('forces', 'World model', 'Forces recon loss, held-out', '{:.3f}', True),
    ('positions', 'World model', 'Positions recon loss, held-out', '{:.5f}', True),
    ('forces_openl', 'World model', 'Open-loop forces loss, held-out', '{:.3f}', True),
    ('kl', 'World model', 'KL posterior-prior, held-out (nats)', '{:.2f}', True),
    ('prior_classes', 'World model', 'Prior classes/var, held-out', '{:.2f}', None),
    ('expert_pos_err', 'World model', 'Position error, expert actions (mm)', '{:.2f}', True),
    ('expert_force_rmse', 'World model', 'Force RMSE, expert actions (counts)', '{:.1f}', True),
    ('wm_motion_ratio', 'World model', 'Hand motion, expert actions (x real)', '{:.2f}', True),
    ('head_forces_rmse', 'Ensemble', 'Mean head open-loop force RMSE (counts)', '{:.1f}', True),
    ('worst_best_forces', 'Ensemble', 'Worst / best head force MSE', '{:.2f}', True),
    ('spread_over_error_forces', 'Ensemble', 'Head spread / error, forces', '{:.3f}', False),
    ('spread_over_error_positions', 'Ensemble', 'Head spread / error, positions', '{:.3f}', False),
    ('head_index_slope_forces', 'Ensemble', 'Log force MSE per head index', '{:+.4f}', None),
    ('ditto_reward', 'DITTO', 'DITTO reward, held-out', '{:.3f}', False),
    ('ditto_action_mse', 'DITTO', 'DITTO action MSE, held-out', '{:.4f}', True),
    ('post_cmd_err', 'DITTO', 'Policy-on-posterior command error (rad)', '{:.3f}', True),
    ('prior_cmd_err', 'DITTO', 'Policy-in-prior command error (rad)', '{:.3f}', True),
    ('policy_pos_err', 'DITTO', 'Policy-in-prior position error (mm)', '{:.2f}', True),
    ('post_jitter', 'DITTO', 'Policy jitter on posterior (x expert)', '{:.2f}', True),
    ('prior_jitter', 'DITTO', 'Policy jitter in prior (x expert)', '{:.2f}', True),
]

BLUE, ORANGE = '#2a78d6', '#eb6834'


def read_jsonl(path: Path) -> list[dict]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_mean(rows, key, last=10):
  values = [float(r[key]) for r in rows if key in r]
  return float(np.mean(values[-last:])) if values else math.nan


def run_row(sweep: Path, seed: int, label: str) -> dict | None:
  seed_dir = sweep / f'seed{seed}'
  setting = seed_dir / label
  if not (setting / 'rollout' / 'done').exists():
    return None
  row = summarize(seed_dir, label)
  ditto = read_jsonl(setting / 'ditto' / 'metrics.jsonl')
  row['ditto_action_mse'] = last_mean(ditto, 'eval/ditto/action_mse')
  rollouts = [json.loads(p.read_text())
              for p in sorted((setting / 'rollout').glob('*/metrics.json'))]
  mean = lambda fn: float(np.mean([fn(m) for m in rollouts]))
  row['expert_force_rmse'] = mean(lambda m: m['expert_actions']['force_rmse_counts'])
  row['policy_pos_err'] = mean(lambda m: m['policy']['position_error_mm'])
  row['prior_cmd_err'] = mean(lambda m: m['policy_action_rmse_rad'])
  heads_path = setting / 'model' / 'ensemble_heads.json'
  heads = json.loads(heads_path.read_text()) if heads_path.exists() else {}
  row['head_forces_rmse'] = (float(np.mean(heads['head_forces_rmse']))
                             if heads else math.nan)
  for key in ('worst_best_forces', 'spread_over_error_forces',
              'spread_over_error_positions', 'head_index_slope_forces'):
    row[key] = heads.get(key, math.nan)
  return row


def stats(values):
  v = np.array([x for x in values if not math.isnan(x)], float)
  if not len(v):
    return math.nan, math.nan, 0
  return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else math.nan, len(v)


def verdict(base, fixed, lower):
  (mb, sb, nb), (mf, sf, nf) = base, fixed
  if math.isnan(mb) or math.isnan(mf):
    return ''
  spread = (0 if math.isnan(sb) else sb) + (0 if math.isnan(sf) else sf)
  if abs(mf - mb) <= spread:
    return 'within seed spread'
  if lower is None:
    return 'higher' if mf > mb else 'lower'
  return 'better' if (mf < mb) == lower else 'worse'


def main():
  parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
  parser.add_argument('--baseline', type=Path, required=True)
  parser.add_argument('--fixed', type=Path, required=True)
  parser.add_argument('--labels', nargs='+', required=True)
  parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
  parser.add_argument('--out', type=Path, required=True)
  args = parser.parse_args()
  args.out.mkdir(parents=True, exist_ok=True)

  # values[variant][label][key] = list over seeds
  values = {}
  for variant, sweep in (('chained', args.baseline), ('independent', args.fixed)):
    values[variant] = {}
    for label in args.labels:
      rows = [r for r in (run_row(sweep, s, label) for s in args.seeds) if r]
      values[variant][label] = {k: [r[k] for r in rows] for k, *_ in METRICS}
      values[variant][label]['seeds'] = len(rows)

  def fmt(spec, mean, std):
    if math.isnan(mean):
      return ''
    return spec.format(mean) + ('' if math.isnan(std) else f' ± {spec.format(std).lstrip("+")}')

  lines, csv_rows = [], []
  for label in args.labels:
    nb = values['chained'][label]['seeds']
    nf = values['independent'][label]['seeds']
    lines += [f'### {label} (seeds: chained {nb}, independent {nf})', '',
              '| Group | Metric | Chained (old) | Independent (fixed) | Change | Verdict |',
              '|---|---|---|---|---|---|']
    for key, group, header, spec, lower in METRICS:
      b = stats(values['chained'][label][key])
      f = stats(values['independent'][label][key])
      change = ''
      if not (math.isnan(b[0]) or math.isnan(f[0]) or b[0] == 0):
        change = f'{100 * (f[0] - b[0]) / abs(b[0]):+.0f}%'
      v = verdict(b, f, lower)
      lines.append(f'| {group} | {header} | {fmt(spec, *b[:2])} | {fmt(spec, *f[:2])} '
                   f'| {change} | {v} |')
      csv_rows.append(dict(label=label, metric=key, chained_mean=b[0], chained_std=b[1],
                           independent_mean=f[0], independent_std=f[1], verdict=v))
    lines.append('')
  (args.out / 'comparison.md').write_text('\n'.join(lines))
  with open(args.out / 'comparison.csv', 'w', newline='') as fh:
    writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0]))
    writer.writeheader()
    writer.writerows(csv_rows)

  cols = 4
  nrows = math.ceil(len(METRICS) / cols)
  fig, axes = plt.subplots(nrows, cols, figsize=(18, 3.9 * nrows), constrained_layout=True)
  for ax in axes.flat[len(METRICS):]:
    ax.set_axis_off()
  x = np.arange(len(args.labels))
  for ax, (key, group, header, _, lower) in zip(axes.flat, METRICS):
    hint = {True: ' (lower better)', False: ' (higher better)', None: ''}[lower]
    ax.set_title(textwrap.fill(f'{group}: {header}{hint}', 46), fontsize=9.5, loc='left', color='#0b0b0b')
    for offset, variant, color in ((-0.12, 'chained', BLUE), (0.12, 'independent', ORANGE)):
      for i, label in enumerate(args.labels):
        seeds = [v for v in values[variant][label][key] if not math.isnan(v)]
        ax.plot([i + offset] * len(seeds), seeds, 'o', color=color, alpha=0.35, ms=5,
                zorder=2)
        mean, std, _ = stats(values[variant][label][key])
        if not math.isnan(mean):
          ax.errorbar(i + offset, mean, yerr=0 if math.isnan(std) else std, fmt='o',
                      color=color, ms=8, capsize=4, lw=2, markeredgecolor='white',
                      zorder=3, label=variant if i == 0 else None)
    ax.set_xticks(x)
    ax.set_xticklabels(args.labels, fontsize=9)
    ax.grid(True, axis='y', color='#e4e3df', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
      ax.spines[side].set_color('#b5b4ae')
    ax.tick_params(colors='#52514e', labelsize=9)
  handles, names = axes.flat[0].get_legend_handles_labels()
  fig.legend(handles, ['Chained heads (old)', 'Independent heads (fixed)'],
             loc='outside upper center', ncol=2, frameon=False, fontsize=11)
  fig.savefig(args.out / 'comparison.png', dpi=90, facecolor='#fcfcfb')
  plt.close(fig)
  print('\n'.join(lines))
  print(f'Wrote {args.out}/comparison.md, comparison.csv, comparison.png')


if __name__ == '__main__':
  main()

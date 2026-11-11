#!/usr/bin/env python3
"""Compare the rates of a run_sweep_hz.sh run: writes summary.md, summary.csv and summary.png.

  python3 eval/mechanical_pen/rate_sweep_summary.py --sweep logdir/rate_sweep

Rates whose runs are missing or unfinished are listed with blanks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

LAST = 10  # Reports averaged at the end of each training run.

# (key, header, format)
COLUMNS = [
    ('hz', 'Rate (Hz)', '{:g}'),
    ('horizon_s', 'DITTO horizon (s)', '{:.2f}'),
    ('wm_forces_loss', 'WM held-out forces loss', '{:.3f}'),
    ('wm_dyn_loss', 'WM held-out dyn loss', '{:.3f}'),
    ('ditto_reward', 'DITTO held-out reward', '{:.3f}'),
    ('ditto_expert_reward', 'Expert-action reward', '{:.3f}'),
    ('ditto_action_mse', 'DITTO held-out action MSE', '{:.4f}'),
    ('episodes', 'Rollouts', '{:d}'),
    ('policy_position_error_mm', 'Policy position error (mm)', '{:.2f}'),
    ('expert_position_error_mm', 'Expert-action position error (mm)', '{:.2f}'),
    ('policy_force_rmse', 'Policy force RMSE', '{:.1f}'),
    ('expert_force_rmse', 'Expert-action force RMSE', '{:.1f}'),
    ('policy_latent_reward', 'Policy latent reward', '{:.3f}'),
    ('policy_command_error_rad', 'Policy command error (rad)', '{:.3f}'),
    ('policy_command_rate', 'Policy command change (/s)', '{:.3f}'),
    ('posterior_policy_command_rate', 'Policy-on-posterior command change (/s)', '{:.3f}'),
    ('posterior_policy_command_error_rad', 'Policy-on-posterior command error (rad)', '{:.3f}'),
    ('expert_command_rate', 'Expert command change (/s)', '{:.4f}'),
    ('policy_motion_mm_s', 'Policy hand motion (mm/s)', '{:.1f}'),
    ('expert_motion_mm_s', 'Expert-action hand motion (mm/s)', '{:.1f}'),
    ('posterior_motion_mm_s', 'Posterior hand motion (mm/s)', '{:.1f}'),
    ('real_motion_mm_s', 'Real hand motion (mm/s)', '{:.1f}'),
]

POLICY = ('Policy in world model', '#2a78d6')
EXPERT = ('Expert actions in world model', '#eb6834')
REAL = ('Real episode', '#6b6a65')
POSTERIOR = ('Posterior (policy on real states)', '#2f9e5a')


def last_mean(path: Path, key: str) -> float:
  if not path.exists():
    return math.nan
  values = []
  for line in path.read_text().splitlines():
    row = json.loads(line)
    if key in row:
      values.append(float(row[key]))
  return float(np.mean(values[-LAST:])) if values else math.nan


def summarize_rate(sweep: Path, hz: int, imag_length: int) -> dict:
  rate_dir = sweep / f'{hz}hz'
  model, ditto = rate_dir / 'model' / 'metrics.jsonl', rate_dir / 'ditto' / 'metrics.jsonl'
  row = {
      'hz': hz,
      'horizon_s': imag_length / hz,
      'wm_forces_loss': last_mean(model, 'eval/forces_local_loss'),
      'wm_dyn_loss': last_mean(model, 'eval/dyn_loss'),
      'ditto_reward': last_mean(ditto, 'eval/ditto/reward'),
      'ditto_expert_reward': last_mean(ditto, 'eval/ditto/expert_reward'),
      'ditto_action_mse': last_mean(ditto, 'eval/ditto/action_mse'),
  }
  rollouts = [json.loads(p.read_text())
              for p in sorted((rate_dir / 'rollout').glob('*/metrics.json'))]
  row['episodes'] = len(rollouts)
  mean = lambda fn: float(np.mean([fn(m) for m in rollouts])) if rollouts else math.nan
  for name, key in (('policy', 'policy'), ('expert', 'expert_actions')):
    row[f'{name}_position_error_mm'] = mean(lambda m: m[key]['position_error_mm'])
    row[f'{name}_force_rmse'] = mean(lambda m: m[key]['force_rmse_counts'])
    # Older rollouts lack the motion metric.
    row[f'{name}_motion_mm_s'] = mean(
        lambda m: m[key].get('position_step_mm', math.nan) * m['hz'])
  row['policy_latent_reward'] = mean(lambda m: m['policy']['latent_reward_mean'])
  row['policy_command_error_rad'] = mean(lambda m: m['policy_action_rmse_rad'])
  # Per-step changes are smaller at higher rates by construction, so compare per second.
  row['policy_command_rate'] = mean(lambda m: m['policy_action_step'] * m['hz'])
  row['expert_command_rate'] = mean(lambda m: m['expert_action_step'] * m['hz'])
  # Older rollouts lack the posterior stream.
  row['posterior_policy_command_rate'] = mean(
      lambda m: m.get('posterior_policy_action_step', math.nan) * m['hz'])
  row['posterior_policy_command_error_rad'] = mean(
      lambda m: m.get('posterior_policy_action_rmse_rad', math.nan))
  row['posterior_motion_mm_s'] = mean(
      lambda m: m.get('posterior', {}).get('position_step_mm', math.nan) * m['hz'])
  row['real_motion_mm_s'] = mean(lambda m: m.get('real_position_step_mm', math.nan) * m['hz'])
  return row


def fmt(value, spec):
  if isinstance(value, float) and math.isnan(value):
    return ''
  return spec.format(value)


def write_tables(rows, sweep: Path):
  with (sweep / 'summary.csv').open('w', newline='') as handle:
    writer = csv.writer(handle)
    writer.writerow([key for key, _, _ in COLUMNS])
    for row in rows:
      writer.writerow([fmt(row[key], spec) for key, _, spec in COLUMNS])
  lines = [
      '# Rate sweep',
      '',
      'Training metrics are means of the last 10 held-out reports. Rollout metrics are means '
      'over the held-out successes rolled out at each rate. "Expert-action" columns replay the '
      "expert's actions through the same world model, so they measure the world model's own "
      'error. The gap between policy and expert-action columns comes from the policy.',
      '',
      'Losses and DITTO rewards come from different world models and step sizes, so compare '
      'them with care across rates. Errors in mm and counts, and changes per second, are '
      'comparable.',
      '',
      '| ' + ' | '.join(header for _, header, _ in COLUMNS) + ' |',
      '|' + '---|' * len(COLUMNS),
  ]
  for row in rows:
    lines.append('| ' + ' | '.join(fmt(row[key], spec) for key, _, spec in COLUMNS) + ' |')
  (sweep / 'summary.md').write_text('\n'.join(lines) + '\n')
  return lines


def plot(rows, sweep: Path):
  hz = np.array([row['hz'] for row in rows], float)
  panels = [
      ('Taxel position error vs real (mm)', [
          ('policy_position_error_mm', POLICY), ('expert_position_error_mm', EXPERT)], False),
      ('Force RMSE vs real (counts)', [
          ('policy_force_rmse', POLICY), ('expert_force_rmse', EXPERT)], False),
      ('Command change per second (normalised)', [
          ('policy_command_rate', POLICY), ('posterior_policy_command_rate', POSTERIOR),
          ('expert_command_rate', REAL)], True),
      ('Decoded hand motion (mm/s)', [
          ('policy_motion_mm_s', POLICY), ('expert_motion_mm_s', EXPERT),
          ('posterior_motion_mm_s', POSTERIOR),
          ('real_motion_mm_s', REAL)], False),
      ('Policy command error (rad RMS)', [
          ('policy_command_error_rad', POLICY),
          ('posterior_policy_command_error_rad', POSTERIOR)], False),
      ('Latent reward over the whole rollout', [('policy_latent_reward', POLICY)], False),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(15, 8.5), constrained_layout=True)
  handles = {}
  for ax, (title, series, log_y) in zip(axes.flat, panels):
    if all(np.isnan([row[key] for row in rows for key, _ in series])):
      ax.set_title(title, fontsize=11, loc='left')
      ax.text(0.5, 0.5, 'No results yet', ha='center', va='center',
              transform=ax.transAxes, color='#57564f')
      ax.set_axis_off()
      continue
    for key, (label, color) in series:
      values = np.array([row[key] for row in rows], float)
      dashed = color == REAL[1]
      (line,) = ax.plot(
          hz, values, color=color, lw=2, ls='--' if dashed else '-', marker='o', ms=8,
          markeredgecolor='white', markeredgewidth=2, label=label)
      handles.setdefault(label, line)
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_xscale('log')
    ax.set_xticks(hz)
    ax.set_xticklabels([f'{h:g}' for h in hz])
    ax.minorticks_off()
    ax.set_xlabel('Rate (Hz)', fontsize=9)
    if log_y:
      ax.set_yscale('log')
    ax.grid(True, color='#e4e3df', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
      ax.spines[side].set_color('#b5b4ae')
    ax.tick_params(colors='#57564f', labelsize=9)
  fig.legend(handles.values(), handles.keys(), loc='outside upper center', ncol=4,
             frameon=False, fontsize=10)
  fig.savefig(sweep / 'summary.png', dpi=90, facecolor='white')
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description='Compare the rates of a rate sweep.')
  parser.add_argument('--sweep', type=Path, default=Path('logdir/rate_sweep'))
  parser.add_argument('--rates', type=int, nargs='*', default=None,
                      help='Default: every <hz>hz folder in the sweep')
  parser.add_argument('--imag-length', type=int, default=15)
  args = parser.parse_args()

  rates = args.rates or sorted(
      int(p.name[:-2]) for p in args.sweep.glob('*hz') if p.name[:-2].isdigit())
  if not rates:
    raise SystemExit(f'No rate folders in {args.sweep}')
  rows = [summarize_rate(args.sweep, hz, args.imag_length) for hz in sorted(rates)]
  lines = write_tables(rows, args.sweep)
  plot(rows, args.sweep)
  print('\n'.join(lines[-(len(rows) + 2):]))
  print(f'Wrote {args.sweep}/summary.md, summary.csv and summary.png')


if __name__ == '__main__':
  main()

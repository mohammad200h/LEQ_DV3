#!/usr/bin/env python3
"""Compare the settings of a run_sweep_posterior.sh run: writes summary.md, summary.csv, summary.png.

  python3 eval/mechanical_pen/posterior_sweep_summary.py --sweep logdir/posterior_sweep_10hz

Settings whose runs are missing or unfinished are listed with blanks. "Held-out" metrics
are the world model's eval/ reports (val_frac episodes); "training" metrics are its
report/ reports on training episodes.
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
import yaml  # noqa: E402

LAST = 10  # Reports averaged at the end of each training run.

# (key, header, format)
COLUMNS = [
    ('name', 'Setting', '{}'),
    ('post_classes', 'Posterior classes/var, held-out', '{:.2f}'),
    ('prior_classes', 'Prior classes/var, held-out', '{:.2f}'),
    ('post_classes_train', 'Posterior classes/var, training', '{:.2f}'),
    ('ent_gap', 'Prior - posterior entropy, held-out (nats)', '{:.2f}'),
    ('kl', 'KL held-out (nats, unclipped)', '{:.2f}'),
    ('kl_train', 'KL training (nats, unclipped)', '{:.2f}'),
    ('forces_train', 'Forces recon, training', '{:.2f}'),
    ('forces', 'Forces recon, held-out', '{:.2f}'),
    ('forces_gap', 'Forces held-out / training', '{:.2f}'),
    ('forces_openl', 'Forces open-loop, held-out', '{:.2f}'),
    ('positions', 'Positions recon, held-out', '{:.4f}'),
    ('ditto_reward', 'DITTO held-out reward', '{:.3f}'),
    ('ditto_expert_reward', 'Expert-action reward', '{:.3f}'),
    ('episodes', 'Rollouts', '{:d}'),
    ('post_cmd_step', 'Policy-on-posterior command change/step', '{:.4f}'),
    ('prior_cmd_step', 'Policy-in-prior command change/step', '{:.4f}'),
    ('expert_cmd_step', 'Expert command change/step', '{:.4f}'),
    ('post_cmd_err', 'Policy-on-posterior command error (rad)', '{:.3f}'),
    ('expert_motion', 'Expert actions in prior, hand motion (mm/s)', '{:.1f}'),
    ('post_motion', 'Posterior hand motion (mm/s)', '{:.1f}'),
    ('real_motion', 'Real hand motion (mm/s)', '{:.1f}'),
    ('expert_pos_err', 'Expert actions in prior, position error (mm)', '{:.2f}'),
]

BLUE, ORANGE, GREEN, GREY = '#2a78d6', '#eb6834', '#2f9e5a', '#6b6a65'

# Plain-language introduction written at the top of summary.md.
GUIDE = """\
## How to read this page

This section explains the words and numbers below for readers new to machine learning.
The results table comes after it.

### What was tested

Two models are trained here. They work as a pair.

- **The world model** learns how the hand behaves from recorded demonstrations. It
  learns what the joint positions and fingertip forces will be after a given command.
  Think of it as a simulator learned from data.
- **The policy** (trained with DITTO) is the part that would drive the hand. It decides
  which joint command to send next. It is trained to copy the successful
  demonstrations, using the world model to try out its own commands.

The world model doesn't keep the hand's state as plain numbers. It squeezes everything
it has seen into a short code called **`z`**. `z` is 32 small dials, and each dial can
point at one of 16 positions (called *classes*). The policy reads `z` to decide what
to do.

The world model can guess `z` in two ways:

- **Prior:** its guess *without* looking at the sensors, from memory alone. The policy
  uses this while it practises inside the world model.
- **Posterior:** its guess *after* looking at the real sensor readings. The policy
  would use this on the real hand.

Neither is a single answer. Each is a spread of chances over the 16 positions of each
dial, and the model picks one position at random according to those chances. If the
chances are spread out, that random pick changes a lot from one step to the next, and
the output jitters. The aim of this sweep was a **more certain posterior**: chances
piled on one position per dial.

### Words used in the table

- **Setting:** one version of the training recipe. `baseline` is the current recipe.
  Each other row changes one or more knobs:
  - `rep0.01`: weaken the pull of the posterior toward the prior (10× weaker).
  - `free0.1`, `free0`: lower or remove an allowance that lets the posterior and prior
    disagree a little at no cost.
  - `forces3`: count mistakes in predicting fingertip forces 3× as heavily.
  - `_wd`: add *weight decay*, which keeps the model's internal numbers small so it
    can't simply memorise the training data.
  - `_noise`: add a little random noise to the force readings the model sees, so it
    learns patterns rather than exact values.
  - `all`: every change above at once.
- **Training vs held-out:** *training* episodes are the demonstrations the model
  learned from. *Held-out* episodes were kept aside and never shown to it during
  training. Held-out numbers are the honest ones: they show how the model does on data
  it hasn't seen, which is what the real hand will give it.
- **Overfitting:** when a model memorises its training data instead of learning the
  general pattern. You can spot it when training numbers look much better than
  held-out numbers.
- **Classes/var (classes per variable):** how many of the 16 positions of each `z` dial
  are still realistic choices on average. **16** means the model has no idea, and
  **1** means it is certain. Lower is more certain.
- **Entropy:** the standard measure of how spread out a set of chances is. *Classes/var*
  is entropy turned into "how many equally likely choices". Entropy is measured in
  *nats*, a unit like bits.
- **KL (KL divergence):** how much the posterior and prior disagree, in nats. 0 means
  they agree exactly. A large held-out KL means that on new data, the model's memory
  (prior) and what it sees (posterior) disagree a lot.
- **Recon (reconstruction loss):** how badly the world model rebuilds the sensor
  readings from `z`. **Lower is better.** Compare these between rows, not as absolute
  amounts. The numbers are on the same scale in every row, even when a setting weighted
  forces more heavily during training.
- **Forces held-out / training:** held-out forces loss divided by training forces loss.
  1 means the model does as well on new data as on old. Bigger means more overfitting.
- **Open-loop:** the world model predicts several steps ahead without seeing the
  sensors, so this tests its memory (the prior). It is always worse than the normal
  loss.
- **DITTO reward:** how closely the policy's imagined motion matches the demonstration,
  from 0 to 1. **Expert-action reward** is the score you get by replaying the
  demonstrator's own commands. It is the best you can realistically expect, so the
  policy's reward should be close to it.
- **Rollouts:** how many held-out demonstrations the policy was tested on.
- **Command change/step:** how much the policy's joint command changes from one step to
  the next. It measures jitter. **Lower is smoother.** The *expert* row is the
  demonstrator's own value; the policy should come close to it.
  - *Policy on posterior:* the policy fed real readings, as it would be on the hand.
  - *Policy in prior:* the policy acting inside the world model's imagination.
- **Command error (rad):** how far the policy's commands are from what the
  demonstrator commanded at the same moment, in radians of joint angle (0.1 rad is about
  6 degrees). **Lower is better.**
- **Hand motion (mm/s):** how fast the fingertips move, averaged over the episode.
  Jittery predictions show up as more movement than the real hand makes, so **closer to
  "Real hand" is better.**
- **Position error (mm):** how far the fingertips end up from where the real hand was,
  when the expert's commands are run through the world model. **Lower is better.** It
  measures how good the world model is as a simulator.

### How to read the results

1. **Start with the baseline row.** Every other row is judged against it.
2. **Did the posterior become more certain?** Look for *posterior classes/var* lower
   than the baseline's. Also check that it is clearly lower than the *prior*: the point
   of looking at the sensors is to be surer than memory alone.
3. **Is it really better, or just memorising?** Check that *forces recon, held-out*
   didn't get worse and that *held-out / training* didn't grow. A model that is more
   certain but wrong on new data is worse, not better.
4. **Did it help the policy?** Look at *policy-on-posterior command change/step*: did it
   move toward the expert value? Then look at *command error*: did the policy get more
   accurate?
5. **Mind the sample size.** Each setting was trained once (one *seed*) and tested on a
   few rollouts. Small differences, say under 10%, can be chance. Before trusting a
   winner, retrain it with a different `SEED` and check that the result holds.

`summary.png` plots the same numbers. Each panel's title says what is measured, and the
dashed grey line in a panel is the reference (training data, the expert, or the real
hand). The GIFs in `<setting>/rollout/<episode>/rollout.gif` show the motion itself.
"""


def read_metrics(path: Path) -> list[dict]:
  if not path.exists():
    return []
  return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def last_mean(rows: list[dict], key: str) -> float:
  values = [float(row[key]) for row in rows if key in row]
  return float(np.mean(values[-LAST:])) if values else math.nan


def stoch_size(model_dir: Path) -> int:
  try:
    config = yaml.safe_load((model_dir / 'config.yaml').read_text())
    return int(config['dyn']['rssm']['stoch'])
  except (OSError, KeyError, TypeError):
    return 32


def summarize(sweep: Path, name: str) -> dict:
  setting = sweep / name
  model = read_metrics(setting / 'model' / 'metrics.jsonl')
  ditto = read_metrics(setting / 'ditto' / 'metrics.jsonl')
  stoch = stoch_size(setting / 'model')
  # Entropy is summed over the z variables; exp(entropy per variable) is the number of
  # equally likely classes it is worth.
  classes = lambda ent: math.exp(ent / stoch) if not math.isnan(ent) else math.nan
  post = last_mean(model, 'eval/post_ent/mean')
  prior = last_mean(model, 'eval/prior_ent/mean')
  row = {
      'name': name,
      'post_classes': classes(post),
      'prior_classes': classes(prior),
      'post_classes_train': classes(last_mean(model, 'report/post_ent/mean')),
      'ent_gap': prior - post,
      'kl': last_mean(model, 'eval/kl/mean'),
      'kl_train': last_mean(model, 'report/kl/mean'),
      'forces_train': last_mean(model, 'report/forces_local_loss'),
      'forces': last_mean(model, 'eval/forces_local_loss'),
      'forces_openl': last_mean(model, 'eval/openl_forces_local_loss'),
      'positions': last_mean(model, 'eval/positions_loss'),
      'ditto_reward': last_mean(ditto, 'eval/ditto/reward'),
      'ditto_expert_reward': last_mean(ditto, 'eval/ditto/expert_reward'),
  }
  # Forces are the losses most likely to overfit; a sharper posterior that raises this
  # ratio is confidently wrong on new episodes. Losses are log-likelihood terms, so the
  # ratio is only meaningful while both are positive.
  row['forces_gap'] = (row['forces'] / row['forces_train']
                       if row['forces_train'] > 0 else math.nan)
  rollouts = [json.loads(p.read_text())
              for p in sorted((setting / 'rollout').glob('*/metrics.json'))]
  row['episodes'] = len(rollouts)
  mean = lambda fn: float(np.mean([fn(m) for m in rollouts])) if rollouts else math.nan
  row['post_cmd_step'] = mean(lambda m: m['posterior_policy_action_step'])
  row['prior_cmd_step'] = mean(lambda m: m['policy_action_step'])
  row['expert_cmd_step'] = mean(lambda m: m['expert_action_step'])
  row['post_cmd_err'] = mean(lambda m: m['posterior_policy_action_rmse_rad'])
  row['expert_motion'] = mean(lambda m: m['expert_actions']['position_step_mm'] * m['hz'])
  row['post_motion'] = mean(lambda m: m['posterior']['position_step_mm'] * m['hz'])
  row['real_motion'] = mean(lambda m: m['real_position_step_mm'] * m['hz'])
  row['expert_pos_err'] = mean(lambda m: m['expert_actions']['position_error_mm'])
  # Rate-free jerkiness: 1 means as smooth as the expert or the real hand.
  row['hz'] = mean(lambda m: m['hz'])
  row['post_jitter'] = mean(
      lambda m: m['posterior_policy_action_step'] / m['expert_action_step'])
  row['prior_jitter'] = mean(lambda m: m['policy_action_step'] / m['expert_action_step'])
  row['wm_motion_ratio'] = mean(
      lambda m: m['expert_actions']['position_step_mm'] / m['real_position_step_mm'])
  row['post_motion_ratio'] = mean(
      lambda m: m['posterior']['position_step_mm'] / m['real_position_step_mm'])
  row['post_cmd_per_s'] = mean(lambda m: m['posterior_policy_action_step'] * m['hz'])
  row['expert_cmd_per_s'] = mean(lambda m: m['expert_action_step'] * m['hz'])
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
      '# Posterior-certainty sweep',
      '',
      *GUIDE.splitlines(),
      '',
      '## Results',
      '',
      'World-model metrics are means of the last 10 reports: held-out is `eval/`, training '
      'is `report/`. Classes/var is exp(entropy / number of z variables): 16 means a '
      'uniform guess, 1 means certain. KL is unclipped (`kl/mean`), so it is comparable '
      'across free-bits settings, unlike `dyn_loss`. Rollout metrics are means over the '
      'held-out successes; "policy on posterior" is the policy queried on real readings, '
      'as on the hardware.',
      '',
      'A setting helps if posterior classes/var falls well below the prior, held-out '
      'forces reconstruction and the held-out/training ratio do not get worse, and the '
      'policy-on-posterior command change falls toward the expert. forces_loss_scale '
      'multiplies the forces loss in training only; the reported losses are unscaled, so '
      'they compare directly.',
      '',
      '| ' + ' | '.join(header for _, header, _ in COLUMNS) + ' |',
      '|' + '---|' * len(COLUMNS),
  ]
  for row in rows:
    lines.append('| ' + ' | '.join(fmt(row[key], spec) for key, _, spec in COLUMNS) + ' |')
  (sweep / 'summary.md').write_text('\n'.join(lines) + '\n')
  return lines


def plot(rows, sweep: Path):
  names = [row['name'] for row in rows]
  x = np.arange(len(names))
  # (title, [(key, label, colour, dashed)])
  panels = [
      ('z classes per variable, held-out (16 = uniform)', [
          ('post_classes', 'Posterior', GREEN, False),
          ('prior_classes', 'Prior', BLUE, False)]),
      ('KL posterior vs prior (nats, unclipped)', [
          ('kl', 'Held-out', BLUE, False),
          ('kl_train', 'Training', GREY, True)]),
      ('Forces reconstruction loss', [
          ('forces', 'Held-out', BLUE, False),
          ('forces_openl', 'Held-out, open-loop', ORANGE, False),
          ('forces_train', 'Training', GREY, True)]),
      ('Command change per step', [
          ('post_cmd_step', 'Policy on posterior', GREEN, False),
          ('prior_cmd_step', 'Policy in prior', BLUE, False),
          ('expert_cmd_step', 'Expert', GREY, True)]),
      ('Decoded hand motion (mm/s)', [
          ('expert_motion', 'Expert actions in prior', ORANGE, False),
          ('post_motion', 'Posterior', GREEN, False),
          ('real_motion', 'Real hand', GREY, True)]),
      ('Policy-on-posterior command error (rad RMS)', [
          ('post_cmd_err', 'Policy on posterior', GREEN, False)]),
  ]
  fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
  for ax, (title, series) in zip(axes.flat, panels):
    ax.set_title(title, fontsize=11, loc='left')
    if all(np.isnan([row[key] for row in rows for key, *_ in series])):
      ax.text(0.5, 0.5, 'No results yet', ha='center', va='center',
              transform=ax.transAxes, color='#57564f')
      ax.set_axis_off()
      continue
    for key, label, color, dashed in series:
      values = np.array([row[key] for row in rows], float)
      ax.plot(x, values, color=color, lw=2 if not dashed else 1.5,
              ls='--' if dashed else '-', marker='o', ms=8,
              markeredgecolor='white', markeredgewidth=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=25, ha='right', fontsize=9)
    ax.grid(True, axis='y', color='#e4e3df', lw=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
      ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
      ax.spines[side].set_color('#b5b4ae')
    ax.tick_params(colors='#57564f', labelsize=9)
    ax.legend(frameon=False, fontsize=9)
  fig.savefig(sweep / 'summary.png', dpi=90, facecolor='white')
  plt.close(fig)


def main():
  parser = argparse.ArgumentParser(description='Compare the settings of a posterior sweep.')
  parser.add_argument('--sweep', type=Path, default=Path('logdir/posterior_sweep_10hz'))
  parser.add_argument('--configs', nargs='*', default=None,
                      help='Default: every setting folder in the sweep, sorted by name')
  args = parser.parse_args()

  names = args.configs or sorted(
      p.name for p in args.sweep.iterdir() if (p / 'model').is_dir())
  if not names:
    raise SystemExit(f'No settings in {args.sweep}')
  rows = [summarize(args.sweep, name) for name in names]
  lines = write_tables(rows, args.sweep)
  plot(rows, args.sweep)
  print('\n'.join(lines[-(len(rows) + 2):]))
  print(f'Wrote {args.sweep}/summary.md, summary.csv and summary.png')


if __name__ == '__main__':
  main()

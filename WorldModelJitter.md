# World-model and policy jitter

Why the DITTO policy and the decoded world-model rollouts on the mechanical pen data look jerky, what has been measured, and what to try next. The DITTO setup is described in `Ditto_DV3_Implementation.md`.

**Summary**

- The world model learns the motion. Fed the expert's actions, its decoded hand stays about 1.6–1.9 mm from the real hand over whole episodes.
- The jitter in the decoded hand is noise around that correct path. It comes from drawing the stochastic latent `z` at random at every imagined step.
- The policy's commands are jerky even on real observations (the posterior), almost as much as in imagination. So imagination is not the cause of the command jitter.
- The posterior is barely more certain than the prior: about 5–6 equally likely classes per variable out of 16, against 5.5–6.5 for the prior ([Judging the posterior and prior](#judging-the-posterior-and-prior)). So on real readings the actor still sees a very random `z`, and that may be what makes the commands jerky. [The test with the most likely z](#open-test-most-likely-z-on-the-posterior) decides between `z` and the actor.

---

## Background: prior and posterior

The RSSM state has two parts:

| Part | What it is | Behaviour |
|---|---|---|
| `h` (`deter`) | continuous memory | smooth, deterministic given its inputs |
| `z` (`stoch`) | 32 categorical variables, 16 classes each | drawn from a distribution at every step |

`z` is drawn from one of two distributions:

- **Prior** (imagination): the model has only `h`, with no sensor reading. DITTO training and the "Policy in world model" and "Expert actions" streams of `rollout_leq_wm.py` use this.
- **Posterior** (observation): the model has `h` plus the current Xela reading. Running the policy on the hardware uses this. `policy()` calls `self.dyn.observe(...)`, [ditto_agent.py:234-251](dreamerv3/ditto_agent.py#L234-L251).

Both draw `z` with `self._dist(logit).sample(...)` ([nets.py:88](dreamerv3/nets.py#L88) for the posterior, [nets.py:136](dreamerv3/nets.py#L136) for the prior). `mode='eval'` changes only the actor, which outputs `tanh(mode)`. It does not change how `z` is drawn.

---

## Judging the posterior and prior

DreamerV3 logs everything needed to judge both networks in `metrics.jsonl`. `report/` keys are training batches; `eval/` keys are the held-out episodes (`val_frac: 0.1`). Losses and entropies are in nats (natural-log units).

`z` has 32 variables of 16 classes, so its entropy runs from 0 (certain) to 32 × ln 16 = **88.7 nats** (every class equally likely). Dividing by 32 and taking `exp` gives the number of equally likely classes per variable.

### What each metric tells you

| Metric | Measures | Good sign |
|---|---|---|
| `post_ent/mean` | Uncertainty about `z` after seeing the reading | Clearly below `prior_ent` |
| `prior_ent/mean` | Uncertainty about `z` before seeing it | Low, but at or above `post_ent` |
| `dyn_loss`, `rep_loss` | KL(posterior ‖ prior): how much the reading adds beyond what the prior predicted. The same value with different stop-gradients; clipped below at 1 nat (free bits, [nets.py:144-152](dreamerv3/nets.py#L144-L152)) | Near 1 on held-out data, close to training |
| `dyn_loss_std` | How bursty that surprise is across steps | Small; large means rare steps the prior misses badly |
| `forces_local_loss`, `positions_loss` | Reconstruction from the posterior | Held-out close to training |
| `openl_forces_local_loss`, `openl_positions_loss` | Open-loop prediction from the prior, after a short observed context | Only somewhat above reconstruction |

How to read them together:

1. **Posterior quality:** held-out reconstruction. A low value means `h` and `z` hold what the sensors show.
2. **Prior quality:** open-loop error against reconstruction. The prior can be no better than the posterior, so the ratio is how much imagination adds.
3. **Generalisation:** held-out against training for both. A large gap means the model fits the training episodes better than new ones.
4. **Randomness of a draw:** posterior entropy. This one matters for jitter, because every draw of `z`, in imagination or on the hardware, is taken from that distribution.

### Results from the rate sweep

Means of the last 10 reports of each world model (50k steps, `logdir/rate_sweep/<hz>hz/model/metrics.jsonl`):

| Rate | Posterior entropy, held-out (classes/var) | Prior entropy, held-out (classes/var) | KL train | KL held-out | KL std held-out | Forces recon train | Forces recon held-out | Forces open-loop held-out | Positions recon held-out |
|---|---|---|---|---|---|---|---|---|---|
| 10 Hz | 51.4 (5.0) | 54.7 (5.5) | 1.71 | 5.91 | 5.97 | 0.78 | 4.07 | 7.80 | 0.003 |
| 20 Hz | 55.7 (5.7) | 58.5 (6.2) | 1.78 | 4.15 | 5.51 | 0.83 | 3.61 | 6.00 | 0.003 |
| 50 Hz | 57.4 (6.0) | 59.6 (6.4) | 1.86 | 2.93 | 5.32 | 1.03 | 3.97 | 5.30 | 0.003 |
| 80 Hz | 56.7 (5.9) | 59.0 (6.3) | 1.86 | 2.78 | 5.30 | 1.01 | 3.98 | 5.06 | 0.003 |
| 100 Hz | 57.6 (6.0) | 59.9 (6.5) | 1.92 | 2.67 | 5.34 | 1.15 | 4.00 | 4.89 | 0.003 |

What this shows:

- **Positions are learned well at every rate.** Held-out reconstruction is 0.003, the same as open-loop.
- **Forces overfit.** Held-out force reconstruction is 3.6–4.1 at every rate, against 0.8–1.2 in training, so the posterior describes new episodes' forces about 4× worse.
- **The prior adds error on new episodes, more so at low rates.** Held-out open-loop forces are 1.9× reconstruction at 10 Hz and 1.2× at 100 Hz, because each step at a low rate covers more time.
- **KL on held-out data is 1.4–3.5× training,** again largest at 10 Hz for the same reason. `dyn_loss_std` of about 5.3–6.0 means the surprise is concentrated in a few steps, probably contact events (not checked).
- **The posterior is barely sharper than the prior.** Seeing the reading removes only 2.2–3.3 nats out of about 55–60, leaving 5–6 equally likely classes per variable. A draw of `z` on real readings is nearly as random as a draw in imagination.

### Why the posterior stays this uncertain

The only pressure on the posterior to be certain comes from the decoder. If `h` already carries most of what the decoder needs, a wide `z` costs almost nothing, and the KL term pulls the posterior toward the wide prior. Training KL sitting just above the 1-nat floor fits this. It also suggests the most likely `z`, or an `h`-only actor, would lose little information. Neither is measured yet.

### Checklist for a world model used for control

- Held-out reconstruction close to training: **fails for forces**, passes for positions.
- Held-out open-loop error only somewhat above reconstruction: **borderline at 10–20 Hz**, passes at 50–100 Hz.
- Held-out KL near training: **fails at 10–20 Hz**, borderline at 50–100 Hz.
- Posterior entropy clearly below prior entropy: **fails at every rate.** This is the check that matters most for smooth commands.

---

## Why the decoded world model is jerky

1. While imagining, the model has no reading to correct it, so at each step it predicts a probability for each class of `z` and draws one at random.
2. That spread is the model's uncertainty, and it is wide: about 5.5–6.5 equally likely classes per variable out of 16 ([Judging the posterior and prior](#judging-the-posterior-and-prior)).
3. Each step's draw is independent. Consecutive steps can land on different classes even when the real hand barely moved, and the decoder turns each class into a different taxel position.
4. Training sees one step at a time. It learns the right per-step probabilities, but nothing penalises jumps between consecutive draws.

The result is a hand that tracks the right path on average but shakes around it.

### Evidence

Held-out episode -1950, 600 steps at 100 Hz, expert actions fed into the world model:

| Latent setting | Hand step (mm/step) | Force step (/step) | High-frequency position jitter (mm) | Position error vs real (mm) |
|---|---|---|---|---|
| Real episode | 0.055 | 0.27 | 0.005 | – |
| Random head, random `z` (current) | 0.363 | 1.15 | 0.112 | 1.9 |
| Fixed head, random `z` | 0.353 | 1.15 | 0.111 | 1.9 |
| Random head, most likely `z` | 0.153 | 0.48 | 0.043 | 2.7 |
| Fixed head, most likely `z` | 0.143 | 0.43 | 0.039 | 3.5 |

- Drawing `z` at random is the main cause. Taking the most likely class cuts the jitter by about 60%.
- Switching between the 7 prior heads adds only about 3%.
- The jitter left with the most likely `z` comes from the argmax flipping between neighbouring classes, because `z` is discrete. The most likely `z` also drifts further from the real path, so it is a viewing aid, not a better model.

Rate sweep, 3 held-out successes per rate, expert actions in the world model:

| Rate | Position error (mm) | Decoded hand motion (mm/s) | Decoded hand motion (mm/step) | Real hand motion (mm/s) |
|---|---|---|---|---|
| 10 Hz | 1.55 | 4.5 | 0.45 | 2.6 |
| 20 Hz | 1.68 | 6.9 | 0.35 | 2.7 |
| 50 Hz | 1.73 | 13.8 | 0.28 | 2.7 |
| 80 Hz | 2.07 | 23.6 | 0.30 | 2.7 |
| 100 Hz | 2.12 | 29.3 | 0.29 | 2.7 |

The decoded hand moves about 0.3–0.45 mm per step at every rate, while the real hand moves 2.7 mm/s. So the jitter is a roughly fixed amount per draw, and its total per second grows with the number of draws per second. Lower rates jitter less per second and track the real hand slightly better.

---

## Is the policy jerky on real observations too?

Every rollout metric reported so far (position error, force error, hand motion, command error and command change) comes from the prior. That is open-loop imagination: after the start step, the model never sees another reading. The hardware uses the posterior, so the policy's actions on posterior latents were scored separately from the saved `rollout.npz` files (`observed_policy_action` against `expert_action`).

Rate sweep, 3 held-out successes per rate, normalised action units:

| Rate | Command change, prior (/s) | Command change, posterior (/s) | Posterior change per step | Expert change (/s) | Command error, prior (RMS) | Command error, posterior (RMS) |
|---|---|---|---|---|---|---|
| 10 Hz | 0.086 | 0.084 | 0.0084 | 0.014 | 0.103 | 0.090 |
| 20 Hz | 0.164 | 0.159 | 0.0080 | 0.014 | 0.091 | 0.068 |
| 50 Hz | 0.461 | 0.434 | 0.0087 | 0.015 | 0.115 | 0.075 |
| 80 Hz | 0.689 | 0.691 | 0.0086 | 0.015 | 0.123 | 0.082 |
| 100 Hz | 1.127 | 0.802 | 0.0080 | 0.015 | 0.225 | 0.070 |

- **Posterior error is flat across rates:** 0.07–0.09 at every rate. On real readings, the policy tracks the expert about equally well at every rate.
- **Prior error grows with rate:** imagination errors compound over more steps at higher rates, most clearly at 100 Hz.
- **The posterior is about as jerky as the prior.** The exception is 100 Hz, where the prior is about 40% worse.
- **The jitter is a fixed amount per decision:** on real readings the policy's command changes by about 0.008 per step at every rate, while the expert's changes by about 0.015 per second. So command jitter per second grows in proportion to the rate: about 6× the expert's at 10 Hz and about 55× at 100 Hz.
- **So imagination noise explains the decoded hand's jitter, but not the command jitter.** The command jitter looks like independent noise added at each decision. That noise could come from the actor, or from the posterior's `z`, which is nearly as random as the prior's (see [Judging the posterior and prior](#judging-the-posterior-and-prior)). Either way, a lower rate, or holding each command for several steps, cuts it directly.

---

## Likely causes of the command jitter

This section is reasoning and has not been tested.

1. **The reward does not reward smoothness.** DITTO rewards matching the expert's latent `h` at each step. A command that wobbles slightly around the right value reaches almost the same `h`, so it earns almost the same reward.
2. **The world model hides small wobbles.** It was trained only on smooth teleoperated commands, and its predictions are already noisy, so a small change in the command barely changes the reward. A jerky command and a smooth one score about the same.
3. **The actor reacts to every step.** Its input latent changes a little at every step, including the random `z`, so its output does too. The expert holds each command for many steps, and nothing teaches the actor to do the same.
4. **The training signal is noisy.** REINFORCE with exploration noise σ of 0.01–0.1 gives noisy updates, and small output jitter never costs reward.

### Open test: most likely `z` on the posterior

The posterior still draws `z` at random, from a distribution with 5–6 equally likely classes per variable, so the command jitter could well come from `z`. The test is to run the posterior with the most likely `z` and compare the command change:

- commands stay jerky: the actor is the cause, so fix the actor;
- commands turn smooth: the latents are the cause, so fix how `z` reaches the actor.

The 10 and 20 Hz checkpoints from the rate sweep are enough for this, with no retraining.

---

## Fixes

### For viewing the world model

- **Average several draws of `z`, or decode the most likely `z`,** in `rollout_leq_wm.py`. This shows the model's mean prediction without sampling noise and needs no retraining.

### For the policy, cheapest first

| Fix | What it does | Cost |
|---|---|---|
| Filter on the hardware | Low-pass or rate-limit commands before they are sent. Sensible for safety anyway. | none |
| Most likely `z` in eval | Make `observe` use the argmax `z` when `mode='eval'`. This matches the expert cache, which already uses argmax ([ditto_agent.py:216](dreamerv3/ditto_agent.py#L216)). | code only |
| Smoothness penalty | Subtract `c·\|a_t − a_{t−1}\|²` from the DITTO reward. Works directly on cause 1. | DITTO retrain |
| Behaviour-cloning term | Add a loss pulling the actor toward the expert's actions on real states. The expert's actions are smooth, so this teaches smoothness directly. | DITTO retrain |
| Previous command as input | Give the actor its previous command, or have it output a change in the command, so holding still is the easy default. | DITTO retrain |
| `h`-only actor, or `z` probabilities | Feed the actor `deter` alone, or `z`'s probabilities instead of a random draw. Removes `z` noise from its input. | DITTO retrain |
| Lower control rate | Fewer independent draws and fewer decisions per second. Already being measured by `eval/mechanical_pen/rate_sweep.sh`. | world-model and DITTO retrain |
| Smoothness loss in the world model | Penalise jumps between consecutive decoded steps. | world-model retrain |

---

## Before running on the hardware

- The policy draws `z` from the posterior at every step, and the posterior is nearly as uncertain as the prior. Expect command jitter close to the posterior numbers above.
- The control loop must run at the policy's training rate, with readings prepared as in training: Xela interpolated to that rate, forces z-scored with `forces_norm`. `preprocess()` does the z-scoring.
- `rollout_leq.py` supports only the squeeze and syringe sims. There is no pen environment or hardware loop yet.

## Reproducing the numbers

- Rate sweep: `./eval/mechanical_pen/rate_sweep.sh`, results in `logdir/rate_sweep/summary.{md,csv,png}`.
- One rollout: `./eval/leap/rollout_leq_wm.sh <GPU> <DITTO logdir> --num-episodes 3 --start <steps>`. `rollout.npz` holds `policy_action` (prior), `observed_policy_action` (posterior) and `expert_action`.
- Posterior and prior metrics: the `report/` and `eval/` keys listed above in each `logdir/rate_sweep/<hz>hz/model/metrics.jsonl`, averaged over the last 10 reports.
- The latent-setting table came from a one-off script on the earlier 100 Hz checkpoint (`logdir/ditto/leap_mechanical_pen/0`). It is not in the repo.

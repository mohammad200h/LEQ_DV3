# DITTO on DreamerV3

DITTO (Dream Imitation) trains an imitation policy inside a frozen world model. The policy is rewarded for keeping its imagined latent state close to the expert's latent state over a short rollout. It needs no environment interaction and no task reward during policy training.

This repo runs DITTO on the DreamerV3 world model that LEQ also uses, on the mechanical pen demonstrations from the Leap hand with Xela skin.

- Paper: [DITTO: Offline Imitation Learning with World Models](https://arxiv.org/abs/2302.03086)
- Released code: [brantondemoss/DITTO](https://github.com/brantondemoss/DITTO) (a DreamerV2 research snapshot; this repo reimplements the objective and does not run that code)

**Status:** training runs end to end and matches the expert closely inside the world model. The policy has not been tested on a task, in simulation or on the hand, and it is still much jittier than the expert. See [Results](#results) and [Known limits](#known-limits).

## Pipeline

```text
compose_playbacks/DV3 chunks (564 episodes, 100 Hz)
        │  data/leap/mechanical_pen is a tracked link to this folder
        ▼
1. World model   --algo model   all episodes    logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt
        ▼
2. Expert cache  frozen model   successes only  encoded once, before training
        ▼
3. DITTO         --algo ditto   frozen model    logdir/ditto/<TASK>/<SEED>/
        ▼
4. Evaluation    report/* on training successes, eval/* on held-out successes
```

### Commands

Train the world model, if you don't already have a checkpoint:

```bash
cd LEQ_DV3
./eval/mechanical_pen/train_model.sh 0 0 leap_mechanical_pen
```

Train DITTO:

```bash
./eval/leap/train_ditto.sh [GPU] [SEED] [TASK] [extra main.py flags...]
# defaults: GPU=0 SEED=0 TASK=leap_mechanical_pen
```

The full 50k run took about 22 minutes on an RTX 4090. For a quick check, append `--run.steps 6000`.

| Environment variable | Default | Purpose |
|---|---|---|
| `MODEL_CHECKPOINT` | `logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt` | World model to freeze |
| `LOGDIR` | `logdir/ditto/<TASK>/<SEED>` | Output folder |
| `SUITE` | `mechanical_pen` | Config block from `configs.yaml` |
| `IMAG_LENGTH` | `15` | Horizon H |
| `DITTO_GAMMA` | `0.95` | Return discount |
| `ACTENT` | `3e-4` | Entropy coefficient |
| `ACTOR_MINSTD`, `ACTOR_MAXSTD` | `0.01`, `0.1` | Actor std range, before tanh |
| `WANDB_ENABLE` | `True` | Set to `False` to keep the run local |

Train at a lower rate. Each rate needs its own world model, because a checkpoint learns dynamics at one step size:

```bash
python3 data/leap/resample_chunks.py --hz 20      # writes data/leap/mechanical_pen_20hz; takes about 15 s
./eval/mechanical_pen/train_model.sh 0 0 leap_mechanical_pen_20hz
./eval/leap/train_ditto.sh 0 0 leap_mechanical_pen_20hz
```

The source chunks aren't changed, and the same 56 episodes are held out at every rate. See [Rate and horizon](#rate-and-horizon).

Watch a policy inside its world model. This needs no simulator or hand:

```bash
./eval/leap/rollout_leq_wm.sh [GPU] [LOGDIR] [extra flags...]
# defaults: GPU=0 LOGDIR=logdir/ditto/leap_mechanical_pen/0
# useful flags: --num-episodes 3, --episode <index or file name>, --start 500, --steps 200,
#               --split train|val|all, --outcome success|fail|any, --no-gif
```

It watches the real episode up to `--start`, then imagines the rest twice from that state: once with the policy choosing every action, and once replaying the expert's actions. The output is in `<LOGDIR>/rollout_wm/<episode>/`:

| File | Contents |
|---|---|
| `rollout.gif` | Real, policy and expert-action hands side by side, from the decoded taxel positions, coloured by decoded force |
| `summary.png` | All 16 joint commands, then force, position, latent reward and command-change curves |
| `metrics.json`, `rollout.npz` | The numbers and arrays behind the plots |

The expert-action stream is the reference: it shows the world model's own error, so only the gap between it and the policy comes from the policy.

## Algorithm as implemented

### Expert latents

`JAXAgent.cache_expert_states` ([jaxagent.py](dreamerv3/jaxagent.py#L476)) runs once after the checkpoint loads. It encodes every expert episode from its true first frame with `encode_segment` ([ditto_agent.py](dreamerv3/ditto_agent.py#L197)) and stores `expert_deter` and `expert_stoch` in the replay chunks. A window starting at frame 200 therefore uses the latent built from the 200 frames before it, as the released featurizer does.

The cache is one sampled posterior per episode. It is recomputed on each start and isn't written to disk. With 181 expert episodes it uses about 2.7 GB of host RAM on top of the observations.

All datasets are created after this step in [train_eval.py](embodied/run/train_eval.py#L119), so no batch is sampled without expert latents.

### Rollouts and alignment

Each sampled sequence has length T (65 in training, 64 in reports). Every start with H frames of expert data ahead of it is used, which gives `T - H + 1` starts ([ditto_agent.py](dreamerv3/ditto_agent.py#L486)): 16 sequences × 51 starts = 816 windows per training batch. These overlap, so they are more correlated than the released code's 512 independent windows.

From each expert start `ŝ₀` the actor rolls out through the frozen prior:

```text
aₖ ~ π(· | ŝₖ)        ŝₖ₊₁ = imagine(ŝₖ, aₖ)        k = 0 .. H-1
```

The state after action `k` is scored against expert frame `k+1`. Replaying the expert's own action at the first step scores 0.9998 against the cached latent, which confirms this alignment.

### Reward

Equation 8 of the paper, applied to the deterministic state `h` only ([ditto_latent_reward](dreamerv3/ditto_agent.py#L52)):

```text
r = (hᴱ · hᵖⁱ) / max(‖hᴱ‖, ‖hᵖⁱ‖)²
```

The last step's reward is 0, as in the released code. The sampled `z` still feeds the RSSM and the actor but isn't compared.

**Why `h` only.** The paper is ambiguous: equation 8 writes `z`, and Algorithm 1 passes `(h, z)`. The released code uses `h` (`state[..., :1024]`). The measurements support that choice, because `z` carries almost no signal here:

| Vector compared | Expert actions | Uniform random actions |
|---|---|---|
| `h` | 0.991 | 0.444 |
| `(h, z)` | 0.668 | 0.339 |
| `z` | 0.185 | 0.180 |

### Returns and losses

With `v` the value of each state ([ditto_agent.py](dreamerv3/ditto_agent.py#L543)):

```text
V^λₖ = rₖ + γ((1-λ) v(ŝₖ₊₁) + λ V^λₖ₊₁)        V^λ_last = v(ŝ_last)        γ = λ = 0.95
adv  = (V^λ - v) / std(V^λ - v), mean-centred across the batch

actor  = mean(-log π(aₖ | ŝₖ) · sg(adv)) - actent · entropy
value  = 0.01 · 0.5 · mean((v - sg(V^λ))²)
```

- The actor is REINFORCE. There is no backprop through the dynamics.
- The world model is frozen because the optimizer only receives the actor and value modules when `run.model_checkpoint` is set.
- There is no continue-head discount, as in the released code.
- The action-conditioned LEQ critic and replay critic aren't used. Their losses are zero placeholders.
- `loss()` still evaluates the world-model terms, so `forces_local_loss` and `dyn_loss` are useful health checks even though they aren't optimized. They do inflate `opt_loss`.

### Deployment

`policy()` returns `tanh(mode)`, which is deterministic. The released code samples.

## Data handling

### Force normalisation

The checkpoint was trained on `symlog((forces - mean) / std)`, so DITTO must use the same scale. `preprocess` applies `jaxutils.norm_forces` ([ditto_agent.py](dreamerv3/ditto_agent.py#L741)), which covers training, the expert cache and `policy()`.

Without it nothing crashes, but the model sees the wrong scale: `forces_local_loss` is about 10,000 instead of about 1, and `dyn_loss` about 11 instead of about 1.9.

**Checkpoint guard:** `train_ditto.sh` exits with an error if `forces_norm` in the checkpoint's `config.yaml` differs from the run's config block.

### Expert selection

The world model trains on all episodes, because failures still teach it what the hand and contact can do. DITTO uses only successes, because its reward pulls the policy toward whatever latents it is given.

[filter_expert_files](dreamerv3/main.py#L223) keeps a file when its final-step reward is at least `ditto_expert_min_reward` (0.5). It runs only for `--algo ditto`, and after the split, so held-out episodes stay held out. If nothing passes, it raises an error. If the demos are reprocessed with different reward values, change this threshold.

| | All episodes | Successes used by DITTO |
|---|---|---|
| Train split | 508 | 169 |
| Held-out split (`val_frac: 0.1`) | 56 | 12 |
| Total | 564 | 181 |

### Rate and horizon

The chunks are at 100 Hz, so H=15 covers only 0.15 s. See [Known limits](#known-limits). At 20 Hz it covers 0.75 s.

The 100 Hz rate comes from the processing step, not the sensors. `sparsh_processing.py` builds a 100 Hz time grid, fits cubic splines to Xela and the joints, and holds the last command at each step. Rates measured from `state_t`, `cmd_t` and `xela_t` in 40 of the 581 raw recordings:

| Stream | Median rate | At 100 Hz in the chunks |
|---|---|---|
| Joint state (`state_t`) | 10.0 Hz (9.0 Hz of changed values) | Mostly spline interpolation |
| Commands (`cmd_t`) | 29.4 Hz | Held between sends |
| Xela (`xela_t`) | 87.2 Hz | Close to the real rate |

`data/leap/resample_chunks.py` keeps every Nth step, so it doesn't undo the spline. In the 20 Hz chunks, about half of the `positions` frames still fall between recorded joint states.

The resampler keeps the action convention. `action[t]` is the command held when the next kept observation arrives, and it matches the raw commands in the 10 episodes checked. Rewards are summed onto kept steps, and the terminal step is always kept, so every episode's return is unchanged. Force stats at 20 Hz match `forces_norm` to within about 0.1%, so the `mechanical_pen` config applies as is.

## Config reference

| Setting | Value | Why |
|---|---|---|
| `--configs mechanical_pen` | | Provides `forces_norm` and `val_frac`, matching the checkpoint |
| `--imag_length` | 15 | Paper's H |
| `--ditto_gamma` | 0.95 | Released code |
| `return_lambda` | 0.95 | Released code; also the config default |
| `--value.dist mse` | | Plain regression output; see below |
| `--actent` | 3e-4 | The released 0.05 is for discrete actions; see below |
| `--actor.minstd`, `--actor.maxstd` | 0.01, 0.1 | Keeps imagined actions close to the data; see below |
| `ditto_expert_min_reward` | 0.5 | Success is +1 and failure -1 in the current chunks |
| `--loss_scales.actor`, `.value` | 1.0 | The `mechanical_pen` block sets these to 0 for world-model training |
| `--loss_scales.reward`, `.critic`, `.replay_critic` | 0.0 | Unused by DITTO |
| `imag_start`, `plan`, `replay.fracs.priority` | `all`, `False`, 0 | Leave as they are; see [Known limits](#known-limits) |

From the `mechanical_pen` block: `opt.lr` 1e-4 with cosine annealing over 50k steps, 1,000 warmup steps, RMS scaler, AGC 0.3, batch 16 × 65.

## Why the settings differ from released DITTO

### Value head

This was a bug in the port, not in LEQ or DreamerV3. The `symexp_twohot` head is standard DreamerV3, and LEQ trains it correctly with cross-entropy ([agent.py](dreamerv3/agent.py#L512)). Released DITTO uses MSE on a linear output. Combining the two is what fails.

The head has 256 bins spaced exponentially out to ±4.85e8, and its value is the probability-weighted mean of those bins. MSE sends its gradient through the bin values:

- At initialisation, dv/dlogit on the top bin is about 1.9e6.
- Raising only that logit by 1e-4 moves the value from 0 to about 190.
- With an error of 15, the MSE gradient on that logit is about 2.8e7, against 0.004 for cross-entropy.

In a measured run the value fell to -7,470 in the first 1,000 steps. `--value.dist mse` gives a plain output and removes the blow-up. Cross-entropy on the original head would probably also work, but it hasn't been tested.

### Action noise

The released settings don't carry over, for three reasons:

- **Coefficient:** 0.05 was used with an 18-way categorical, whose entropy is capped at about 2.9. On 16 Gaussian log-stds it pushes each one up at a constant rate until `maxstd` stops it.
- **Definition:** this agent uses the entropy before tanh, which ignores saturation. LEQ uses `-logpi`, which includes the tanh correction.
- **Scale:** the DreamerV3-style range of 0.1 to 1.0 starts at σ≈0.89. Even the 0.1 floor equals the data's per-joint spread, and it is hundreds of times larger than the expert's per-step change.

Noisy actions hurt in several ways. They sit outside what the world model saw during training, they swamp the reward (0.92 at σ=0.1 and 0.67 at σ=0.3), and they make REINFORCE noisier. The policy uses its mode at deployment, so this noise isn't sent to the hand.

### Effect of each fix

Each row adds one change to the row above. These are exploratory runs: 40 episodes including failures, 6,000 steps, z-scored forces, scored on training windows.

| Run | Reward (policy mode) | Action MSE | Final σ | Lowest early value |
|---|---|---|---|---|
| Released-style settings | 0.813 | 0.528 | 1.00 | -7,470 |
| + `--value.dist mse` | 0.925 | 0.215 | 0.47 | 4.7 |
| + entropy 3e-4 | 0.977 | 0.020 | 0.106 (at the floor) | 4.7 |
| + std range 0.01–0.1 | 0.986 | 0.007 | 0.022 | 4.8 |
| Expert actions replayed | 0.991 | n/a | n/a | n/a |

The changes were cumulative, so their effects aren't fully separated.

## Comparison with the paper and released code

| Area | Paper / released code | This implementation |
|---|---|---|
| World model | DreamerV2 RSSM, `h` 1024, 32×32 categorical | DreamerV3 block GRU, `h` 2048, 32×16 categorical, 7 chained prior heads |
| Observations | Atari pixels | Taxel forces and positions (368 × 3 each) |
| Expert data | Separate from world-model data | Successes only, selected from the world-model data |
| Expert latents | Encoded from the episode start | Same |
| Reward | Eq. 8; code compares `h` | Same, on `h` |
| Horizon | H=15 | H=15, but only 0.15 s at 100 Hz |
| Actions | 18-way categorical | 16-D tanh-normal |
| λ-return, advantage normalisation | γ=λ=0.95, advantage / std | Same |
| Value target | Separate target critic, refreshed every 200 steps | Online value with stop-gradient |
| Value output | Linear, MSE | Linear (`--value.dist mse`), MSE |
| Entropy coefficient | 0.05 hardcoded (the config lists an unused 0.003) | 3e-4 |
| Actor and value size | 8 layers × 256 | 3 layers × 256 |
| Optimizer | Adam, 3e-4 | RMS scaler, 1e-4 cosine, AGC 0.3 |
| Batch | 512 windows | 816 overlapping windows from 16 sequences |
| Deployment | Samples actions | Mode |
| Evaluation | Real environment | Latent metrics on held-out demos only |
| Baselines | D-BC and D-GAIL train alongside | None |

**What this comparison rests on.** The released-code column was checked against the source. The paper's main sections were read through Section 4.2, but its appendix (actor equation and hyperparameter table) was not, so hyperparameters come from the released code.

## Results

**Full run:** 50k steps, the 169 training successes, on the `leap_mechanical_pen` checkpoint. Numbers are means of the last 10 reports. Each report uses one batch, so individual values are noisy. The run is in `logdir/ditto/leap_mechanical_pen/0/`.

| Metric | Training successes (`report/`) | Held-out successes (`eval/`, 12 episodes) |
|---|---|---|
| `ditto/reward` | 0.992 | 0.991 |
| `ditto/expert_reward` | 0.993 | 0.993 |
| `ditto/action_mse` | 0.0057 | 0.0055 |
| `ditto/action_step` | 0.0081 | 0.0082 |
| `ditto/expert_action_step` | 0.0001 | 0.0001 |
| `forces_local_loss` | 0.98 | 1.06 |
| `dyn_loss` | 1.88 | 1.86 |
| `val/mean` | 19.8 | 19.8 |

How to read these:

- **Imitation in the model is close.** Held-out reward nearly reaches the expert reference, and there is no gap between training and held-out episodes.
- **Action error is moderate.** An MSE of 0.0055 is an RMS error of about 0.074 in normalised units, or about 0.15 rad (8.5°) per joint.
- **The policy is jittery.** It moves about 80× more per step than the expert, even with the deterministic mode.
- **Held-out means held out from DITTO only.** The checkpoint's config has no `val_frac`, so the world model trained on all 564 episodes, including these 12. This gives no test of world-model generalisation, and 12 episodes is a small sample.
- **None of this measures task success.**

**Whole-episode rollouts in the world model.** The same 100 Hz policy, using `rollout_leq_wm.sh` defaults on the first 3 held-out successes, observing 8 steps and then imagining 18–22 s. Training only scores 0.15 s windows, so this is a much harder test.

| Metric | Policy | Expert actions (reference) |
|---|---|---|
| Latent reward, mean | 0.39–0.53 | 0.81–0.92 |
| Force RMSE (counts) | 58–82 | 15–39 |
| Taxel position error (mm) | 6.8–10.5 | 1.8–1.9 |
| Command error (rad RMS) | 0.25–0.28 | n/a |
| Command change per step (normalised) | 0.0085–0.0097 | 0.00014–0.00015 |

- **The world model holds up, and the policy drifts.** With expert actions, position error stays near 2 mm for the whole episode. With the policy it is 4–6× larger.
- **The 0.99 window score doesn't carry over to long rollouts.** Latent reward falls to about 0.4–0.5 over the full episode.
- **In the episode inspected (`-1950`), the policy misses the timing of the task.** The expert holds the thumb joints (12–15) still until about 13 s, while the imagined policy starts moving them within the first few seconds. Joints 8–11, which the expert holds still, start well away from the expert's values and then drift.
- **Don't use the reward head as a success signal here.** Its prediction for the end of the episode stays near 0 even with expert actions on a successful episode (below 0.01 in the episode inspected).

## Metrics guide

Metrics are logged under `train/`, `report/` (training successes) and `eval/` (held-out successes).

| Metric | Meaning | Healthy value |
|---|---|---|
| `ditto/reward` | Eq. 8 along sampled imagined rollouts | Rises toward `ditto/expert_reward` |
| `ditto/expert_reward` | Same score from replaying expert actions through the prior. Reports only | About 0.993; this is the practical ceiling |
| `ditto/latent_distance` | `1 - ditto/reward` | Falls |
| `ditto/action_mse` | Policy mode vs expert action, at the same step in the imagined rollout | About 0.005 in the full run |
| `ditto/action_step` | Mean per-step change of the policy mode | Should approach the next row. Currently about 80× higher |
| `ditto/expert_action_step` | Mean per-step change of expert actions | About 0.0001 at 100 Hz, 0.001 at 10 Hz |
| `forces_local_loss` | Frozen world-model reconstruction | About 1. **Near 10,000 means the force scale is wrong** |
| `dyn_loss` | Frozen world-model KL | About 1.9. **Near 11 means the inputs don't match the model** |
| `val/min`, `val/mean` | Value predictions | Positive from the start, rising to about 19.8. **Large negative values mean the value head has blown up** |
| `ent/action/mean` | Actor entropy before tanh | Falls to about -44 at σ≈0.02. **22.7 means σ is pinned at 1.0** |

`opt_loss` includes the frozen world-model terms, so don't read it as the DITTO loss. `eval/*` is meaningful only when `val_frac > 0`. Otherwise it comes from the stub env with zero expert latents, and `eval/ditto/*` means nothing.

## Known limits

- **Command jitter.** The policy's command changes about 9× more per step than the expert's, even on real observations. A reward term `−c·Σⱼ(aₖ,ⱼ − aₖ₋₁,ⱼ)²` was tried and removed. At `c` = 10 (10 Hz, 6,000 steps, one seed) it made commands about 3× smoother but raised the command error by about 50% over whole episodes. An exponential moving average on the commands (α ≈ 0.25–0.35 at 10 Hz) gave the same smoothness with no loss of accuracy in an offline comparison. Filter the commands on the hardware instead. See [WorldModelJitter.md](WorldModelJitter.md).
- **Horizon and jitter.** H=15 at 100 Hz is 0.15 s. Holding the first action for the whole window scores 0.9915, the same as the real expert actions at 0.9911, and action noise up to 0.01 is undetected. The reward therefore can't see motion inside a window, which likely explains the jitter. The likely fix is to train at a lower rate with `resample_chunks.py`. The 20 Hz data is built and a 300-step world-model run confirmed that it loads and trains, but no full world model or DITTO policy has been trained on it yet.
- **No task evaluation.** `rollout_leq.py` supports squeeze and cyringe environments only, and no mechanical pen environment was found. `embodied/envs/leap.py` is a stub with zero observations. `rollout_leq_wm.sh` shows what the world model predicts, not what would happen on the hand. It has only been run on DITTO checkpoints, not LEQ.
- **REINFORCE in 16-D.** The reward is differentiable through the frozen model, so backprop through dynamics may work better. Untested.
- **Chained prior heads.** See [PossibleBug.md](eval/mechanical_pen/PossibleBug.md). All imagination goes through these heads, and fixing them means retraining the world model.
- **No target network.** The value target uses the online value. Training is stable so far, but this differs from the released code.
- **No baselines.** D-BC and D-GAIL aren't implemented. D-BC would be a useful sanity check.
- **Mechanical pen data only.** `_segment_batch` in [jaxagent.py](dreamerv3/jaxagent.py#L589) hardcodes the `forces_local` and `positions` keys, and the expert cache and new metrics assume a continuous `action` key.
- **Settings to leave alone.** `imag_start` must stay `all`. `plan: True` reaches a code path with an undefined variable. Priority replay with `priosignal: td` or `all` doesn't work, because the imagination losses are scalars.

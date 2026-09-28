# DITTO on DreamerV3

DITTO (Dream Imitation) trains an imitation policy inside a frozen world model. The policy is rewarded for staying close to the expert's deterministic state over several imagined steps. This repo already trains a DreamerV3 world model and then a separate actor-critic. DITTO uses that same split. The world model stays DreamerV3. The agent is the LEQ actor-critic with a different imagination reward, a fixed discount, shorter aligned rollouts, and no replay critic.

Paper: [DITTO: Offline Imitation Learning with World Models](https://arxiv.org/abs/2302.03086). The public [DITTO repo](https://github.com/brantondemoss/DITTO) is a DreamerV2 research snapshot and is not what this code runs.

## What was added

| Piece | Role |
| --- | --- |
| `dreamerv3/ditto_agent.py` | Copy of the LEQ agent (`dreamerv3/agent.py`) with the DITTO reward |
| `dreamerv3/main.py` | `--algo ditto` loads that agent |
| `dreamerv3/configs.yaml` | `ditto_gamma` (default 0.95) |
| `eval/leap/train_ditto.sh` | Stage-2 launcher, same pattern as `train_leq.sh` |

Stage 1 is unchanged: `eval/leap/train_model.sh` still trains `--algo model` and writes `logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt`.

## Steps

1. Collect Leap episodes and convert them the same way as LEQ (`data/leap/convert_episodes.py`). DITTO does not read the environment reward during policy learning, but the chunks still need `action`, `forces_local`, and `positions`.
2. Train the world model:

   ```bash
   cd /workspace/LEQ_DV3
   ./eval/leap/train_model.sh 0 0 leap_squeeze
   ```

   The Leap config trains the RSSM, encoder, decoder, reward head, and continue head. DITTO freezes all of those. Imagination uses the encoder and RSSM. The return also multiplies by the frozen continue head. The reward head is still evaluated inside the world-model loss, then multiplied by `--loss_scales.reward 0.0`. It is not an optimizer target and it is not the imagination reward.
3. Train DITTO on that checkpoint:

   ```bash
   ./eval/leap/train_ditto.sh 0 0 leap_squeeze
   ```

   Optional: `MODEL_CHECKPOINT=...`, `IMAG_LENGTH=15`, `DITTO_GAMMA=0.95`, `ACTENT=3e-4`.
4. Checkpoint lands in `logdir/ditto/<TASK>/<SEED>/`.
5. Roll out in the real simulator with the existing script. It rebuilds the agent from the saved `config.yaml`, which has `algo: ditto`:

   ```bash
   ./eval/leap/run_rollout.sh 0 logdir/ditto/leap_squeeze/0
   ```

   `train_eval` episode scores during training are still the stub Leap env, not LeapFlexEnv. Judge the policy with `run_rollout.sh`.

## How the reward is computed

On each sampled sequence the RSSM encodes the demonstration with `observe`, so `replay_outs['deter']` is the expert's deterministic state. Replay marks the first step of every sequence `is_first`, which resets the RSSM there. A start is any time `t` that still has `imag_length` future steps inside that sequence. `configs.yaml` leaves `imag_length` at 5 and `batch_length` at 65. `train_ditto.sh` overrides the horizon to 15, which leaves `65 - 15 = 50` starts. The actor is unrolled from the encoded state at `t` for `H` steps with `imagine`. The reward drops index 0, the shared start, and compares `deter` after each learner action with the expert `deter` at that same future index:

```text
z = deter
r = dot(z_expert, z_learner) / max(||z_expert||, ||z_learner||)^2
```

That is equation 8 of the paper, applied the way the released trainer does it: `ac_trainer.py` calls `max_cos` on `state[..., :1024]`, the deterministic recurrent state `h`, and does not include the sampled categorical `z`. Here `deter` is that state (length 2048 on the Leap `size12m` model; `stoch` stays 32 and `classes` becomes 16). `stoch` is still sampled inside the RSSM on every imagination step and is still an actor input. It is not part of the compared vector. `r = 1` when the two `deter` vectors are equal. Training logs `ditto/reward` and `ditto/latent_distance` (`1 - r`).

The return, critic, value net, and REINFORCE actor are the ones already in `agent.py`, with `return_lambda` 0.95. The script passes `--expectile 0.5`, so the LEQ advantage weight is 0.5 on both sides instead of the yaml default 0.1. The critic loss still adds `slowreg` (1.0) toward the slow target. The value net is trained to match the critic, and that value net is the actor baseline. There is no replay-critic regression on `data['reward']`: `losses['replay_critic']` is a zero placeholder, and the script also sets `--loss_scales.replay_critic 0.0`. Without the model checkpoint, `train()` would optimize the world model too; `train_ditto.sh` always passes one.

When `--run.model_checkpoint` is set, the optimizer updates only actor, critic, and value. Encoder, RSSM, decoder, reward head, and continue head stay at the world-model checkpoint.

## Differences from the paper

These are intentional adaptations, not drop-in DreamerV2 behavior.

- **RSSM is DreamerV3**, the `size12m` Leap model (`deter` 2048, `stoch` 32, `classes` 16, block GRU). The released code used DreamerV2 (`h` 1024, 32 categoricals of rank 32, stored as a 1024-d one-hot block that the reward ignored).
- **Reward vector is `deter` only**, matching `state[..., :1024]` in the released trainer. The actor still conditions on `deter` and `stoch`. Sampled one-hots do not enter the dot product.
- **Discount** is `ditto_gamma` (0.95, the paper value) times the world model's predicted continue probability. The paper used a fixed γ and no learned continue head.
- **Actor-critic** is the LEQ one (two-hot critic, separate value net, percentile return normalization, `slowreg: 1.0`), not the paper's MSE critic. The actor term is still REINFORCE. `--expectile 0.5` turns the asymmetric LEQ weight into a constant 0.5. The yaml default expectile is 0.1 if the script flag is dropped.
- **Entropy coefficient** defaults to `3e-4`, the DreamerV3 continuous-control scale. The paper's `0.05` was for discrete Atari. Set `ACTENT=0.05` to use the paper number; on 16-D `tanh_normal` actions that bonus is likely too large.
- **Windows are cut at the end of each sampled sequence**, not at the on-disk episode file. With the script, the last 15 steps of each length-65 sequence are never imagination starts. Converted Leap episodes do not chain (`succ` is a zero id), so a sequence also does not run into the next episode. The paper sampled windows from full episodes.
- **`imag_start` must stay `all`.** `last` has no future expert state in the batch.
- **`plan: False`.** The planning branch in `policy()` still scores trajectories with the reward head and is not the DITTO objective.

## Missing

- No run of this agent has been executed here. The reward was checked on dummy vectors: identical `deter` scores 1 even when `stoch` differs, and orthogonal `deter` scores 0. A full training step was not run.
- No held-out latent-distance evaluator of the kind in the paper's Figure 2. The training log reports distance on the same imagined batch the optimizer just used.
- No D-GAIL or D-BC baselines.
- Priority replay is off (`replay.fracs.priority` is 0, `priosignal` is `model`). Turning priority on with `priosignal: td` or `all` reshapes the imagination actor/critic loss onto `(batch, batch_length)` and fails. `priosignal: model` uses the full-sequence world-model losses and does not hit that reshape.
- Real Leap return is not logged during `train_eval`. Use `eval/leap/rollout_leq.py` / `run_rollout.sh`.
- The world-model reward head can still be trained in stage 1 (`leap` sets `loss_scales.reward: 1.0`). The imagination reward does not read it. You do not need to retrain the world model with a reward head for DITTO, and a checkpoint trained with `loss_scales.reward: 0.0` is valid as long as the RSSM and continue head were trained.
- Episode termination inside a sequence is handled by the learned continue head, not by splicing `is_terminal` from the demonstration into the latent reward.
- The original DITTO code release is pre-release DreamerV2 and was not ported. This is a reimplementation of the objective on the DreamerV3 agent that LEQ already trains.

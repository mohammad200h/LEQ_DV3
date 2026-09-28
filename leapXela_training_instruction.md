# Leap+XELA training instructions

Two-stage LEQ pipeline on Leap+XELA tactile data (same idea as V-D4RL):

1. **World model** (`--algo model`) — RSSM + encoder/decoder on `forces_local` and `positions`.
2. **LEQ** (`--algo leq`) — freeze the world model, train actor / critic / value with imagined rollouts.

Repos:

| Role | Path |
|------|------|
| Collect demos | `/workspace/sparsh-skin-sim` |
| Convert + train | `/workspace/LEQ_DV3` |

Task name is `leap_<folder>`, e.g. `leap_squeeze` reads `data/leap/squeeze/*.npz`.

---

## 0. Prerequisites

- GPU, JAX, and the `LEQ_DV3` Python env (`pip install -e .` from `LEQ_DV3`).
- W&B (optional but on by default). Put `WANDB_API_KEY` in `/workspace/LEQ/docker/.env` or the environment.
  - Override: `WANDB_PROJECT`, `WANDB_NAME`, or `WANDB_ENABLE=False`.
- Do **not** reuse a world-model checkpoint trained on all-zero rewards for LEQ. After you collect reward data, reconvert and retrain the WM.

---

## 1. Collect episodes

From `sparsh-skin-sim`:

```bash
cd /workspace/sparsh-skin-sim
python data_collection/data_collection.py --config data_collection/config/squeeze.yaml
```

Configs:

- `data_collection/config/squeeze.yaml` — squeeze grasp, default `data/squeeze`
- `data_collection/config/finger_close.yaml` — finger close, default `data/finger_close`

Useful YAML knobs:

- `collection.episodes` — number of demos
- `collection.duration` — max seconds per episode
- `collection.output_dir` — where `episode_XXXX.npz` is written
- `env.num_envs` — parallel collection
- `policy.grasp_pattern` — `squeeze` or `finger_close`

Each `episode_XXXX.npz` is ARS (`action_t`, `obs_{t+1}`, `reward_{t+1}`) and includes:

- `actions` — `(T, 16)` joint targets
- `rewards` — `(T,)` squeeze contact reward from `LeapFlexEnv.step` (0.1 per fingertip in contact)
- `flex_taxel_fk/forces_local` — `(T, 368, 3)`
- `flex_taxel_fk/positions` — `(T, 368, 3)`
- `metadata_json`

Episodes collected before reward logging have no `rewards` key and must be re-collected. Episodes must be **≥ 65 steps** (DreamerV3 `batch_length`). Squeeze runs are typically ~570 steps.

---

## 2. Convert to DreamerV3 chunks

From `LEQ_DV3`:

```bash
cd /workspace/LEQ_DV3
python data/leap/convert_episodes.py \
  --input /workspace/sparsh-skin-sim/data/squeeze \
  --output data/leap/squeeze
```

`--output` folder name **is** the task suffix (`squeeze` → `--task leap_squeeze`).

Conversion:

- Shifts actions ARS → RSA (same as V-D4RL).
- Scales joint targets from `[ACTION_LOW, ACTION_HIGH]` to `[-1, 1]`.
- Writes `forces_local`, `positions`, `action`, `reward`, `is_first`, `is_last`, `is_terminal`.
- Copies `rewards` from the collection episode (aligned with observations; not shifted). Falls back to zeros and warns if the key is missing.

---

## 3. Train the world model

```bash
cd /workspace/LEQ_DV3
./eval/leap/train_model.sh [GPU] [SEED] [TASK]
# defaults: GPU=0  SEED=0  TASK=leap_squeeze
```

Examples:

```bash
./eval/leap/train_model.sh 0 0 leap_squeeze
./eval/leap/train_model.sh 1 0 leap_finger_close
```

This runs `--algo model` and writes:

```
logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt
```

e.g. `logdir/model/leap_squeeze/0/0/checkpoint.ckpt`.

Leap config (`dreamerv3/configs.yaml` → `leap:`) currently zeros actor / critic / value **and** reward:

```yaml
loss_scales.reward: 0.0
loss_scales.actor: 0.0
loss_scales.critic: 0.0
loss_scales.value: 0.0
```

That is correct only for a reconstruction-only WM. **Once rewards exist, train the reward head:**

```bash
CUDA_VISIBLE_DEVICES=0 python3 dreamerv3/main.py \
  --logdir logdir/model/leap_squeeze/0/0 \
  --configs leap \
  --task leap_squeeze \
  --algo model \
  --seed 0 \
  --loss_scales.reward 1.0
```

(or add that override to `train_model.sh`). Leave actor / critic / value at `0.0` in this stage.

LEQ later **freezes** encoder, RSSM, decoder, **and** the reward head. If the reward head is untrained, imagined returns are noise.

Watch `train/rep_loss`, `train/dyn_loss`, and decoder losses on W&B / `metrics.jsonl`.

---

## 4. Visualize the world model

```bash
cd /workspace/LEQ_DV3
./eval/leap/run_visualizer.sh \
  logdir/model/leap_squeeze/0/0 \
  data/leap/squeeze \
  7860
```

Open `http://localhost:7860`. Check:

- Teacher-forced reconstructions of taxel forces / positions
- Open-loop imagination from a chosen timestep

If imagination diverges immediately, do not start LEQ — collect more data or train the WM longer.

---

## 5. Train LEQ

```bash
cd /workspace/LEQ_DV3
./eval/leap/train_leq.sh [GPU] [SEED] [TASK] [EXPECTILE]
# defaults: GPU=0  SEED=0  TASK=leap_squeeze  EXPECTILE=0.1
```

Examples:

```bash
./eval/leap/train_leq.sh 0 0 leap_squeeze
./eval/leap/train_leq.sh 0 1 leap_squeeze          # matches WM seed 1
MODEL_CHECKPOINT=logdir/model/leap_squeeze/0/1/checkpoint.ckpt \
  ./eval/leap/train_leq.sh 0 0 leap_squeeze        # different LEQ seed
```

What this does:

- Loads `--algo leq` (`dreamerv3/agent.py`)
- Restores WM weights from `--run.model_checkpoint`
- Optimizes **only** actor / critic / value (`ac_modules`)
- Imagines `imag_length=5` steps and uses expectile-weighted λ-returns (`expectile=0.1`)

Checkpoint:

```
logdir/leq/<TASK>/<EXPECTILE>/<SEED>/checkpoint.ckpt
```

Do not pass `--algo model` here. Do not start LEQ until:

- Replay chunks contain a non-zero reward
- The WM was trained with `--loss_scales.reward 1.0`

DITTO is a separate stage-2 agent on the same frozen world model. It does not use the reward head. See `Ditto_DV3_Implementation.md` and:

```bash
./eval/leap/train_ditto.sh 0 0 leap_squeeze
```

---

## 6. Roll out the trained LEQ policy in LeapFlexEnv

`embodied/envs/leap.py` is still a **space stub** (zeros), so `train_eval` episode scores are not real Leap performance. Use the rollout script instead — it loads the LEQ checkpoint and steps the real `sparsh-skin-sim` `LeapFlexEnv`.

```bash
cd /workspace/LEQ_DV3
./eval/leap/run_rollout.sh [GPU] [LOGDIR] [ENV_CONFIG] [EPISODES] [--render]
# defaults:
#   GPU=0
#   LOGDIR=logdir/leq/leap_squeeze/0.1/1
#   ENV_CONFIG=/workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml
#   EPISODES=1
```

Examples:

```bash
# Headless, one episode
./eval/leap/run_rollout.sh 0 logdir/leq/leap_squeeze/0.1/1

# With MuJoCo viewer (single env; needs a display / X11)
./eval/leap/run_rollout.sh 0 logdir/leq/leap_squeeze/0.1/1 \
  /workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml 1 --render

# Or call Python directly
CUDA_VISIBLE_DEVICES=0 python3 eval/leap/rollout_leq.py \
  --logdir logdir/leq/leap_squeeze/0.1/1 \
  --env-config /workspace/sparsh-skin-sim/data_collection/config/squeeze.yaml \
  --episodes 5 \
  --render
```

What this does:

- Loads `config.yaml` + `checkpoint.ckpt` from the LEQ logdir
- Builds **one** `LeapFlexEnv` from the collection YAML (ignores `env.num_envs`; always single-env)
- Maps `flex_taxel_fk/{forces_local,positions}` → Dreamer obs
- Runs `agent.policy(..., mode='eval')`
- Denormalizes actions from `[-1, 1]` back to joint ctrl targets
- Prints per-step and total squeeze reward

Useful flags (`rollout_leq.py`):

- `--render` — open MuJoCo passive viewer; sync each step at sim rate
- `--max-steps N` — override episode length (default from collection `duration`)
- `--seed 0` — env reset seed (increments per episode)
- `--sparsh-root /workspace/sparsh-skin-sim` — if the sim repo is elsewhere

**Startup note:** agent load compiles JAX train/report graphs first (~1–2 min). The viewer opens **after** that finishes. Do not Ctrl+C during the shape / “Tracing train function” spam — wait until you see `Logdir:` / `Episode 1/1`.

**Reward:** squeeze reward is `0.1` per fingertip in contact with the object (up to `~0.4`/step). Judge success by total / mean reward, not by the stub `episode/score` from training.

Requires `sparsh-skin-sim` deps in the same Python env (`pytorch_kinematics`, etc.).

---

## Checklist

After you finish reward collection:

- [ ] Re-collect `episode_*.npz` so each file includes `rewards` of shape `(T,)`
- [ ] Re-run conversion into `data/leap/<task>/` and confirm printed reward max is not 0
- [ ] Retrain WM with `--loss_scales.reward 1.0` (new logdir, or wipe the old one)
- [ ] Visualize reconstructions + imagination
- [ ] Train LEQ with `--algo leq` and `--run.model_checkpoint …/checkpoint.ckpt`
- [ ] Roll out in LeapFlexEnv (`./eval/leap/run_rollout.sh … [--render]`) and check contact reward

---

## Paths cheat sheet

```
sparsh-skin-sim/data_collection/config/squeeze.yaml
sparsh-skin-sim/data/squeeze/episode_XXXX.npz          # raw collection

LEQ_DV3/data/leap/convert_episodes.py
LEQ_DV3/data/leap/squeeze/*.npz                        # DreamerV3 chunks
LEQ_DV3/eval/leap/train_model.sh
LEQ_DV3/eval/leap/train_leq.sh
LEQ_DV3/eval/leap/run_visualizer.sh
LEQ_DV3/eval/leap/run_rollout.sh
LEQ_DV3/eval/leap/rollout_leq.py
LEQ_DV3/dreamerv3/configs.yaml                         # leap: block
LEQ_DV3/logdir/model/leap_squeeze/0/0/checkpoint.ckpt  # stage 1
LEQ_DV3/logdir/leq/leap_squeeze/0.1/0/checkpoint.ckpt  # stage 2
```

V-D4RL reference (same two-stage idea, different data):

```
./eval/vd4rl/test_whole.sh dmc_vision dmc_walker_walk-medium_expert model 0 0 {SEED}
./eval/vd4rl/test.sh       dmc_vision dmc_walker_walk-medium_expert leq   0.1 0 {SEED}
```

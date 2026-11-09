# Mechanical pen (Leap + XELA, real hand)

Two-stage LEQ pipeline on the real-hand mechanical pen demos, same as `eval/leap/`:

1. **World model** (`--algo model`): RSSM + encoder/decoder on `forces_local` and `positions`.
2. **LEQ** (`--algo leq`): freeze the world model, then train the actor, critic and value on imagined rollouts.

All commands run from `LEQ_DV3/`.

## Data

The demos are already DreamerV3 chunks, so **don't** run `data/leap/convert_episodes.py` on them.

```
data/leap/mechanical_pen -> /workspace/LeapXELA_Hardware_ws/ros_ws/demonstrations/compose_playbacks/DV3
```

- 564 episodes recorded at 100 Hz, roughly 1100–2750 steps each.
- Each `*.npz` has:
  - `forces_local` `(T, 368, 3)`
  - `positions` `(T, 368, 3)`
  - `action` `(T, 16)`, with joint targets scaled from `[-1.5, 2.5]` to `[-1, 1]`
  - `reward` in `[-1, 0]`
  - `is_first`, `is_last`, `is_terminal`
- `DV3/manifest.json` maps each source episode (`th_*`) to its chunk file.

To use new demos, regenerate `DV3/`. The symlink picks up the new files automatically.

## Config

- Config block: `mechanical_pen:` in `dreamerv3/configs.yaml`. It starts as a copy of `leap:`, so you can tune it separately.
- Task: `leap_mechanical_pen`. The `leap_` prefix makes `main.py` load `data/leap/mechanical_pen/*.npz` and use the `embodied.envs.leap:Leap` stub env.
- Force normalization: `forces_norm` z-scores `forces_local` per axis inside the agent; `positions` are not normalized yet. See [NORMALIZATION.md](NORMALIZATION.md) for why and how.
- Validation split: `val_frac: 0.1` holds out 10% of the demo episodes (56 of 564). The split uses a fixed seed, so the world model and LEQ runs hold out the same episodes. The held-out filenames are written to `<logdir>/val_files.txt`. Set `--val_frac 0` to train on every episode.

## Metrics

- `train/*`: losses on the training batches.
- `report/*`: reconstruction and open-loop prediction on a training batch.
- `eval/*`: the same metrics as `report/*`, on a batch from the held-out episodes. Compare `eval/*` with `report/*` to check for overfitting.
- `episode/*` and `epstats/*` come from the stub env, which returns zeros, so ignore them.

## Train

```bash
# 1. World model -> logdir/model/leap_mechanical_pen/0/<SEED>/
./eval/mechanical_pen/train_model.sh [GPU] [SEED] [TASK]

# 2. LEQ -> logdir/leq/leap_mechanical_pen/<EXPECTILE>/<SEED>/
./eval/mechanical_pen/train_leq.sh [GPU] [SEED] [TASK] [EXPECTILE]
```

- Defaults: `GPU=0`, `SEED=0`, `TASK=leap_mechanical_pen`, `EXPECTILE=0.1`.
- `train_leq.sh` loads `logdir/model/<TASK>/0/<SEED>/checkpoint.ckpt` by default. To use a different checkpoint, set `MODEL_CHECKPOINT`.
- Training resumes from any checkpoint already in the logdir. If that logdir has a run finished before the validation split was added, move it or delete it first. Otherwise the run resumes at the final step and stops right away.
- W&B is on by default. The scripts read `WANDB_API_KEY` from `/workspace/LeapXELA_Hardware_ws/LEQ_DV3/.env` or the environment. You can override `WANDB_PROJECT` or `WANDB_NAME`, or set `WANDB_ENABLE=False` to turn it off.

## Visualize

```bash
./eval/mechanical_pen/run_visualizer.sh [LOGDIR] [DATA] [PORT]
# defaults: logdir/model/leap_mechanical_pen/0/0  data/leap/mechanical_pen  7860
```

This opens a Gradio app at `http://localhost:<PORT>`. Pick a demo episode and a timestep to see the world model's teacher-forced reconstructions and open-loop predictions next to the recorded taxel data. The visualizer lists every episode, so to check generalization, pick episodes from `<logdir>/val_files.txt`.

## No rollout

`eval/leap/` includes `rollout_leq.py`, which runs the policy in the sparsh-skin-sim simulator. There's no simulated mechanical pen env, so this folder doesn't include it. To evaluate the policy, run it on the real hand.

## Explore the data

```bash
./eval/mechanical_pen/run_visualize_data.sh [DATA] [PORT] [LOGDIR]
# defaults: data/leap/mechanical_pen  7861  logdir/model/leap_mechanical_pen/0/0
```

This opens a Gradio app at `http://localhost:<PORT>` for browsing the training demos. It needs no model or GPU. It has tabs for:

- an overview of the dataset and the train/val split,
- a frame-by-frame episode viewer,
- force and position distributions,
- action statistics,
- success vs fail contact curves,
- the PCA coverage map.

Its statistics are cached in `.cache/` after the first run (about 40 s). LOGDIR is used only for `val_files.txt`.

For a guided course through both visualizers and the data, start at [course/README.md](course/README.md).

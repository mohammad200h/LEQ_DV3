# Course: understanding the mechanical pen data and world model

This course is for someone new to data science. It covers:

- what the robot recorded,
- the handful of statistics ideas you need to read the plots,
- the two tools in this folder, page by page,
- what this particular dataset contains, including some surprises worth knowing before you trust a trained model.

Every number quoted in the chapters was measured on the real data (564 episodes in `data/leap/mechanical_pen`, created 2026-10-29). If the data is regenerated, rerun the explorer: the overview page and the numbers boxes update themselves.

## The two tools

| Script | What it shows | Needs |
|---|---|---|
| `./eval/mechanical_pen/run_visualize_data.sh` | The **data**: episodes, forces, actions, rewards, statistics | Only the `.npz` files (CPU) |
| `./eval/mechanical_pen/run_visualizer.sh` | The trained **world model**: what it reconstructs and predicts | A checkpoint + GPU |

Run both from `LEQ_DV3/`:

```bash
./eval/mechanical_pen/run_visualize_data.sh          # http://localhost:7861
./eval/mechanical_pen/run_visualizer.sh              # http://localhost:7860
```

The data explorer's first start reads all 5 GB of episodes once to build its statistics (about 40 s), then caches them in `eval/mechanical_pen/.cache/`. Later starts take a few seconds.

Arguments, all optional: `run_visualize_data.sh [DATA] [PORT] [LOGDIR]`. `LOGDIR` is used only to find `val_files.txt`, so the explorer can tag the held-out episodes.

## Chapters

Read them in order. Each one ends with something to try in the tools.

1. [Meet the data](01_meet_the_data.md): what one episode file contains, and what each number physically means.
2. [The data science toolkit](02_data_science_toolkit.md): histograms, mean/std, percentiles, log scales, z-score, symlog, correlation, PCA, train/val. Just enough to read every plot.
3. [Tour of the data explorer](03_data_explorer_tour.md): every tab and every panel of `run_visualize_data.sh`.
4. [The world model in plain words](04_world_model_in_plain_words.md): encoder, latent state, decoder, teacher forcing, open loop, ensembles.
5. [Tour of the model visualizer](05_model_visualizer_tour.md): every panel of `run_visualizer.sh`, plus what good and bad look like.
6. [What this dataset tells us](06_findings.md): the findings, the caveats and suggested next steps.
7. [Exercises](07_exercises.md): hands-on questions, with answers.

There is also a [glossary](glossary.md) to look up terms as you go.

## Related notes in this folder

- [../README.md](../README.md): how to train.
- [../NORMALIZATION.md](../NORMALIZATION.md): why forces are z-scored. Chapter 2 is the gentle version of it.
- [../PossibleBug.md](../PossibleBug.md): the ensemble prior issue mentioned in chapter 5.

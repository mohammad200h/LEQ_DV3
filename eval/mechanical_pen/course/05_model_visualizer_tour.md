# 5. Tour of the model visualizer

```bash
./eval/mechanical_pen/run_visualizer.sh [LOGDIR] [DATA] [PORT]   # then open http://localhost:7860
```

It loads the checkpoint in `LOGDIR` (default `logdir/model/leap_mechanical_pen/0/0`), so it needs a GPU. Startup compiles the model, so give it a minute.

## Controls

| control | meaning |
|---|---|
| **Episode** | The chunk file (long name). The data explorer's Episode tab shows the chunk name under "Values at this step", so you can copy it across. |
| **Field** | Show *Forces* or *Positions* in the spatial panels. |
| **Window start** | The first step of the 64-step window the model runs on. The model starts with empty memory here. |
| **Step in window** | Which of the 64 steps (0–63) the spatial panels show. The episode step is `window start + step in window`. |
| **Density feature space / radius** | Settings of the data density panel (below). |

Each change re-runs the model on the window.

**Choosing a window.** Most of any episode is "holding the pen still". To see interesting dynamics, use the data explorer first:

- find the step where something happens, for example a thumb press in the contact timeline;
- set Window start to about 20 steps before it, so the event falls inside the open-loop part (after step 8).

For the ending, use `Window start = length − 64`.

## Panel 1: Spatial predictions (2×3 grid)

Each scatter is the hand from the **top (x, y)**, one dot per taxel, coloured by |force| in counts. In *Positions* mode, colour is the distance of the taxel from the hand base origin, which is a much less intuitive quantity. The data explorer's Episode tab shows the same top view next to front and side views, which helps you find your way.

| | | |
|---|---|---|
| **Ground truth**: the recorded data at this step | **Teacher-forced recon**: the model's drawing while seeing the data | **Open-loop imagine**: the model's drawing after imagining from step 8 |
| **\|recon − gt\|**: where recon is wrong | **\|openl − gt\|**: where imagination is wrong | **Sequence error**: MSE over the whole window |

The top three share one colour scale, so compare brightness directly. The error maps (blue → red, coolwarm) have their own scale; check the colour bar before panicking.

What to look for:

- **Recon ≈ ground truth.** If even recon misses a contact, the encoder or decoder can't represent it. Often the reason is that it's rare: check the density panel.
- **Open loop keeps the contact pattern.** Right after step 8 it should look almost like recon, and it drifts later. A good model keeps *which* patches are pressed even if it gets the magnitude a bit wrong.
- **Open-loop N/A** for steps before 8 is expected: there is no imagination yet.

**Sequence error (bottom right):**

- Solid lines: recon MSE for forces and positions.
- Dashed lines: open-loop MSE.
- Dotted vertical line: the current step.

Typical shape: open-loop error starts near recon error at step 8 and rises. Forces and positions are in different units (counts² and m²), so compare each one only with itself. For forces, √MSE is the typical error in counts. Compare it with the size of the contacts you see in the colour bar.

## Panel 2: Reward & continuation

Black is the truth, the solid colour is recon and dashed is open loop.

- **Reward**: 0 everywhere except the last step of an episode. In almost every window the true curve is flat 0, and the useful check is that predictions stay near 0. To see the interesting case, choose `Window start = length − 64`, so the ±1 lands at step 63.
- **Continuation**: 1 everywhere except the last step of *success* episodes, where it drops to 0. Fail episodes keep cont = 1 even at the end, because they were truncated, not finished.

The task signal sits on 564 single steps out of 1.27 million, so expect the model to be uncertain about rewards.

## Panel 3: Data density around the current step

This is the PCA map from chapter 2.7, the same as the data explorer's Coverage tab, focused on the current step.

- **Left**: the whole dataset, with a white line for this episode, a red line for the window and a red star for the current step.
- **Middle**: a zoom around the star. The dashed circle is the search radius. Orange dots are dataset steps inside it from **other episodes**; blue dots are from the **same** episode.
- **Right**: the number of dataset steps within the radius, for every step of the window. Black counts all data; orange counts other episodes only.

How to use it:

- **Many orange neighbours**: the model has seen many similar states from other demos, so predictions should be reliable.
- **Only blue or few neighbours**: the state is rare, and only this episode visits it. Expect larger errors, especially open loop. If this is a *train* episode, the model may simply have **memorised** it.
- When an error spike in the Sequence error panel lines up with a dip in the orange curve, the model is failing because of **missing data**, not because of a bug.

The radius is a fraction of the dataset's overall spread: 0.1 means 10% of the typical distance between a point and the dataset mean, measured in 8 PCA dimensions. Increase it if every count is 0.

## Panel 4: Open loop per ensemble prior

The same open-loop rollout is repeated 7 times, each time **forcing one head** (prior 0…6).

- **Grid**: each head's drawing at the current step, next to the ground truth. The title gives that head's MSE.
- **Bottom left**: each head's MSE over the window, plus their mean (thick black) and the "sampled" rollout (dashed) from Panel 1.
- **Bottom right**: mean |force| over the hand for each head, against the truth (thick black). The grey band is the spread between heads. The red dashed line marks where open loop starts (step 8).

Reading it:

- **A narrow grey band that follows the black line**: confident and right.
- **A wide band**: uncertain. Usually it lines up with low data density.
- **A narrow band away from the black line**: confidently wrong. This is the dangerous case for LEQ, because the uncertainty penalty won't catch it.
- **One head that always drifts**: a weak head. PossibleBug.md suggests later heads (deeper in the chain) may drift more.

## Panel 5: Metrics (text)

The numbers at the selected step:

- recon and open-loop MSE for forces and positions,
- reward and cont, true vs recon,
- the density summary: how many neighbours, the share of the dataset they make up, and how many other episodes they come from.

## A routine for judging a checkpoint

1. Pick 3–5 **val** episodes (from the explorer's table, `split = val`) and 3–5 **train** episodes. Remember the caveat in chapter 2.9: the current checkpoint may have seen the val episodes.
2. For each, look at three windows:
   - early (`Window start = 0`, fingers closing),
   - mid-episode (holding),
   - the end (`length − 64`).
3. Note the force recon √MSE and the open-loop √MSE at step 63.
4. If val is clearly worse than train, the model is overfitting.
5. Where errors are large, check the density panel. Low support means you need more data like that; high support means the model needs to improve.
6. Look at the ending of a few **fail** episodes. Chapter 6 explains why the last 0.15 s of those is special.

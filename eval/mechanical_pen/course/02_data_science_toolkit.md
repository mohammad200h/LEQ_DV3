# 2. The data science toolkit

These are the ideas behind every plot in both tools. Each section ends with **where you see it**.

## 2.1 Distribution and histogram

A **distribution** describes which values a quantity takes and how often. A **histogram** draws it: split the value range into bins, count how many samples land in each bin, and draw a bar per bin.

- A tall narrow spike means the value almost always sits there. An example is the ring finger joints, which never move.
- A wide hump means the value varies a lot.
- Several separate bumps (a *multimodal* distribution) usually mean several distinct situations. For example, a joint target that sits at "open" in some episodes and at "grip" in others.

The explorer draws some histograms as **lines** rather than bars so three axes (x, y, z) can share one plot. They are normalised to a *density*: the area under each curve is 1, so curves built from different amounts of data are comparable.

*Where:* Tactile stats (top row, positions); Actions (16 small plots); Overview (episode lengths).

## 2.2 Mean, standard deviation, median, percentiles

- **Mean**: the average. Sensitive to extreme values.
- **Standard deviation (std, σ)**: the typical distance from the mean. Most data lies within a few σ of the mean.
- **Median**: the middle value; half the data is below it. Robust to extreme values.
- **Percentile**: the *p*-th percentile has *p*% of the data below it. The median is the 50th. The 25th and 75th percentiles bound the **middle 50%** of the data (the "interquartile range").

Forces show why the median matters. The mean taxel |F| is pulled up by rare hard presses (up to about 8,000 counts), while the median is 47.

*Where:* Success vs fail draws the median line with a 25–75% band; violin plots mark the median; the numbers box shows mean and std.

## 2.3 Log scales

When values span several orders of magnitude (1, 10, 100, 1,000), a normal axis squashes the small ones flat. A **log axis** gives each factor of 10 the same space.

- On a log y-axis, a straight descending line is an exponential decay. Long flat "tails" mean rare but very large values exist.
- `symlog` axes are log scales that also handle zero and negative values. They are linear near 0 and logarithmic further out.

*Where:* Tactile stats (y-axes of the force histograms); Coverage (colour = log count); model visualizer density curve.

## 2.4 z-score (standardisation)

```
z = (x - mean) / std
```

This rewrites each value as "how many standard deviations from the mean". After it, the data has mean 0 and std 1, whatever its original units. It is a *linear* change: twice as far stays twice as far.

The world model does this to forces before anything else, with the numbers in `configs.yaml` (`forces_norm`): mean (6.1, 8.5, 28.9) and std (140, 95, 146) counts for x, y and z. The explorer recomputes them from every timestep and gets the same numbers. That confirms the config matches the data.

Positions are **not** z-scored yet. See [NORMALIZATION.md](../NORMALIZATION.md).

*Where:* Tactile stats, panel 2.

## 2.5 symlog

```
symlog(x) = sign(x) * log(1 + |x|)
```

- Near zero it barely changes the value (`symlog(0.5) ≈ 0.41`).
- For large values it compresses hard (`symlog(1000) ≈ 6.9`).

DreamerV3 applies it to every vector input and predicts in symlog space. That stops a rare 3,000-count spike from dominating the training loss.

The catch: if values are all large, symlog turns *everything* into a log, and a hard press barely differs from a light one. That's why the forces are z-scored first. After that, **87.8%** of force values have |z| < 1, the region where symlog is almost the identity (the yellow band in Tactile stats, panel 3).

## 2.6 Correlation

The **correlation** of two quantities is a number from −1 to +1:

- **+1**: they always move together (one goes up, so does the other).
- **0**: no linear relationship.
- **−1**: they move in opposite directions.

A **correlation matrix** shows every pair as a coloured square. Red means positive, blue negative, white near zero. The diagonal is always +1, since each quantity is perfectly correlated with itself.

Correlation isn't causation. Two joints can be correlated simply because the human operator moved them together.

*Where:* Actions tab, bottom right.

## 2.7 PCA: squeezing many numbers into two

One timestep of observation is 368 × 3 forces plus 368 × 3 positions: 2,208 numbers. You can't plot 2,208 dimensions.

**PCA (principal component analysis)** finds the directions in that space along which the data varies the most.

- **PC1** is the direction of largest variation.
- **PC2** is the next largest, at right angles to PC1, and so on.

Projecting every timestep onto PC1 and PC2 gives a 2D "map" where:

- similar states land close together,
- the most important differences are spread out.

The text under the plot says how much information the map keeps. For example, "the first 8 PCs keep 86.2% of the variance": 8 numbers describe 86% of the variation of the 2,208 observation numbers. The map draws only 2 of those 8, so points that look close on the map can still be far apart in the other 6 PCs.

*Where:* Coverage tab; the model visualizer's "Data density" panel.

## 2.8 Density and coverage

Colour the PCA map by **how many timesteps fall in each small square**, on a log scale. Bright areas are visited constantly; dark or empty areas were rarely or never seen.

This matters because a learned model is only reliable **where it has seen data**. The model visualizer counts how many dataset timesteps are within a radius of the current one, and how many of those come from *other* episodes. Few neighbours means the model is extrapolating there, so expect bigger errors.

## 2.9 Train / validation split and overfitting

Hold back some episodes the model **never trains on** (the *validation* or *val* set). Then compare its errors there to its errors on training episodes.

- Similar errors: the model **generalises**.
- Much lower error on train than on val: the model **overfits**. It memorised the training episodes rather than learning how the hand works.

Here `val_frac: 0.1` holds out 56 of 564 episodes, always the same ones (fixed seed). The explorer tags each episode `train` or `val`.

Two caveats for this repo:

- The current checkpoint in `logdir/model/leap_mechanical_pen/0/0` has no `val_frac` in its `config.yaml`. It was trained before the split existed, so **it has seen every episode**, and "val" episodes are not truly unseen for it.
- The val set is only 21% success (12 of 56), compared with 33% (169 of 508) in train. Small random splits are often unbalanced like this.

## 2.10 Aligning time series

Episodes have different lengths, so "step 500" means different things in different episodes. To compare many episodes, line them up at a **meaningful event** and look at time relative to it. The explorer aligns at the **end** of the episode (where the click happens, or the cut-off) or at the **start**. Then it shows the median and the middle 50% across episodes at each relative time.

*Where:* Success vs fail tab.

## 2.11 MSE and RMSE

**Mean squared error** = the average of (prediction − truth)². It is in *squared* units (counts², m²), which are hard to picture. Take the square root to get the **RMSE**, which is back in normal units. For example:

- force MSE 2,500 counts² → RMSE 50 counts,
- position MSE 1e-6 m² → RMSE 1 mm.

Only compare MSEs measured in the same units. A position MSE of 1e-6 looks tiny next to a force MSE of 2,500 only because metres are big and positions barely change.

*Where:* everywhere in the model visualizer.

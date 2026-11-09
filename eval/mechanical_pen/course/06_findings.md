# 6. What this dataset tells us

These are the findings from exploring the 564 demos (data of 2026-10-29). Each one names the place in the explorer where you can see it yourself. They're ordered by how much they could affect the world model and LEQ.

## Things to act on

### 6.1 Failed episodes end with the hand letting go, a "tell" for the −1 reward

In **91% of fail episodes** (348 of 383), at least one joint command jumps by more than 0.1 rad in the last 30 steps. The last such jump comes a median **15 steps (0.15 s)** before the end. Only **1 of 181** success episodes does this. As the hand releases, the index, middle and palm forces drop.

*See:* Success vs fail tab (the sudden drop at the right edge of index, middle and palm); Episode tab on any fail episode → *Jump to last step*.

**Why it matters.** The reward head can learn "the hand releases ⇒ −1" instead of "the thumb didn't click ⇒ −1". For LEQ, the cheapest way to avoid the −1 is then "never release", not "click the pen". The fail label effectively lands on the wrong behaviour.

**Possible fix.** Trim the last ~20–30 steps of fail episodes when building the chunks in `dv3_processing.py`, or end the episode at the last step before the command jump. Then rebuild `DV3/` and retrain.

### 6.2 Episode length almost gives away the outcome

90% of successes are 15–22 s long; 90% of fails are 20–27 s. Fails run until the recording is cut off.

*See:* Overview, left histogram.

The world model never sees time directly, so it can't use this. Any model given time, a step counter, or long enough context could, and anyone comparing success against fail statistics should keep it in mind. Per-step averages over whole episodes mix in "how long the hand sat there".

### 6.3 The task signal is tiny

There is exactly one non-zero reward per episode: 564 out of 1,268,291 steps (0.04%). A 65-step training window contains it only if it covers the episode's last step, about 3% of the windows that can be drawn. The reward and continuation heads learn from very little.

*See:* Overview bullets; Episode tab, reward panel.

### 6.4 Some joints never move, so the model knows nothing about moving them

- The ring finger's 4 joints are constant: 100% of steps unchanged, a single histogram spike.
- `if_rot` and `mf_rot` barely move.

*See:* Actions tab.

If LEQ's policy outputs different values for these joints, the world model is **extrapolating**, and its predictions there mean nothing. Consider fixing these actions to their recorded values (or masking them) when training LEQ.

### 6.5 Validation caveats

- The current checkpoint (`logdir/model/leap_mechanical_pen/0/0`) was trained before `val_frac` existed. Its `config.yaml` has no `val_frac` and there is no `val_files.txt`, so it has seen all 564 episodes. To measure generalisation, retrain. The new run writes `val_files.txt`; pass the same logdir to `run_visualize_data.sh` and the explorer will use it.
- The fixed split is unbalanced: 21% success in val (12/56) against 33% in train (169/508). Val metrics are dominated by fail episodes, and their endings are affected by 6.1.

## Things to keep in mind

### 6.6 "No contact" is not zero force

Taxels have resting levels of tens of counts: a median of 47 counts over all taxels and steps, and about 33 counts on the ring finger, which never touches anything.

*See:* Tactile stats → *Typical (median) |force|* map; Overview → ring violin.

When looking for contact, look at the **rise above an episode's starting level**, not at absolute values.

### 6.7 Positions are nearly static and not normalised

Taxels move a median 0.022 mm per step. Forces vary about 230× more than positions per dimension (after the forces' z-score and symlog). So the position part of the loss is tiny whether or not the model tracks the fingers.

*See:* Tactile stats, bottom-middle panel. Details in [NORMALIZATION.md](../NORMALIZATION.md).

### 6.8 The last action of every episode is a filler

The converter sets the final action to 0 rad (scaled −0.25) for all 16 joints. The resulting jump is not a real command. It's one step per episode, so it's harmless for training, but don't read meaning into it in the plots.

### 6.9 Actions are step-like

85–95% of steps repeat the previous command, because commands arrive slower than the 100 Hz sample rate. The model sees a staircase of actions with forces responding smoothly.

*See:* Actions tab, smoothness panel.

### 6.10 Data quality checks pass

- no NaN or inf values,
- no taxel reads exactly 0,
- every episode has exactly one non-zero reward,
- success ⇔ terminal holds,
- the force mean and std computed from the data match `forces_norm` in `configs.yaml`.

The one extreme episode, with a peak |F| of about 8,300 counts and the top dot in the Overview scatter, is worth a look in the Episode tab.

### 6.11 Where the pen is held

The index finger's `link2`/`link1A`, the middle finger's `link5A` and the thumb's `link15` are each above 100 counts about 90% or more of the time. The fingertips are rarely loaded. Most of the 368 taxels carry little information for this task: 312 ever exceed 100 counts, and only 76 ever exceed 1,000.

*See:* Tactile stats → taxel activity map.

## The success signature

Across episodes, the thing that separates success from fail is the **thumb**. In successful episodes the thumb's force rises steadily over the last 2–3 s as it pushes the button. In failed ones it stays lower and drops about 1.5 s before the end. Every episode starts the same way (align at start: the curves overlap). This is the behaviour the world model has to capture, and LEQ has to reproduce, for the task to work.

*See:* Success vs fail tab, thumb panel.

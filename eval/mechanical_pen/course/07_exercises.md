# 7. Exercises

Try each one in the tools before opening the answer.

## Data explorer

**E1. Read a shape.** How many force numbers does one 2,000-step episode hold? How many bytes is that as float32?

<details><summary>Answer</summary>

2,000 × 368 × 3 = 2,208,000 numbers × 4 bytes ≈ 8.8 MB. Positions add the same again. That's why the dataset is 5 GB even compressed.
</details>

**E2. Your first episode.** Open the Episode tab with `th_1_1`. At which step does the thumb start pressing? Which patch carries it?

<details><summary>Answer</summary>

In the contact timeline, the `thumb link15` row brightens from about step 1,040–1,060 (t ≈ 10.5 s). In the per-finger panel the red thumb curve rises at the same point. It's `link15`, the thumb's middle link, not the tip.
</details>

**E3. Fixed vs rescaled colours.** Go to step 0 of any episode and switch between the two colour modes. Why does "Rescale every step" make the hand look busy even though nothing is touching?

<details><summary>Answer</summary>

Rescaling stretches whatever values exist to the full colour range. At rest, the taxels' resting levels (tens of counts) fill the whole scale, so noise looks like contact. Use "Fixed per episode" to judge *how much*, and "Rescale" only to see *where*.
</details>

**E4. Find an outlier.** In the Overview table, sort by `max |F|`. Open the top episode and jump to its strongest contact. Is it one taxel or a whole patch? Real contact or a glitch?

<details><summary>Answer</summary>

Look at the hand image and the patch bars. If one taxel is far brighter than its neighbours and it only lasts a step or two in the timeline, it's likely a sensor spike. If a whole patch rises and falls smoothly, it's a real hard press. Either way, symlog keeps a single spike from dominating the loss.
</details>

**E5. The normalisation chain.** In Tactile stats, what share of z-scored force values fall where symlog is nearly linear? Why do we want this to be high?

<details><summary>Answer</summary>

87.8%. In that region symlog doesn't distort values, so twice the force stays about twice as big for the model. Only the rare extreme values get compressed. Without the z-score, only about 3% would be in the linear region. Every force would be on a log scale, and a hard press would look much like a light touch.
</details>

**E6. Threshold game.** On the taxel map, raise the threshold from 100 to 1,000. Which patches still light up? What does that tell you about the grasp?

<details><summary>Answer</summary>

Mainly the index finger's middle links (`link2`, `link1A`), the middle finger's `link5A` and the thumb's `link15`. The pen is squeezed between the middle sections of the fingers and the thumb, not held with the fingertips.
</details>

**E7. Correlation.** Name two joints with strong positive correlation, and one joint whose correlation is meaningless.

<details><summary>Answer</summary>

Strong positive pairs include `if_mcp` with `if_pip`, `if_mcp` with `mf_mcp`, `mf_mcp` with `mf_pip`, and `th_cmc` with `th_axl`. Any ring joint (`rf_*`) is meaningless: it is constant, and correlation measures how *variations* line up.
</details>

**E8. Success vs fail.** Align at the end. Which finger separates success from fail? Now align at the start: what changes, and what does that tell you?

<details><summary>Answer</summary>

The thumb separates them at the end. At the start every curve overlaps: success and fail episodes begin the same way, and the difference is only in what the thumb does at the end. That's the behaviour that decides the reward.
</details>

**E9. The release tell.** Open five fail episodes and *Jump to last step*. What happens in the actions and forces in the last ~20 steps? Why could this be a problem for LEQ?

<details><summary>Answer</summary>

The joint targets move (the hand starts releasing) and the index, middle and palm forces drop, about 0.15 s before the end. Because the −1 reward comes right after, the reward model can associate −1 with "releasing" instead of "not clicking". See 6.1.
</details>

**E10. Coverage.** On the Coverage map (Observation, split by Train / val), are there regions only val episodes visit? What would it mean if there were?

<details><summary>Answer</summary>

Val should sit inside the train cloud. A region only val visits would be a behaviour the model never trained on: high val error there would mean "never seen this", not "overfitting". Use the model visualizer's density panel on such steps; "other episodes" counts will be low.
</details>

## Model visualizer

**M1. Recon vs open loop.** Pick any episode, Window start 0, Field = Forces. Move *Step in window* from 0 to 63. When does the open-loop panel appear, and how does its error compare with recon over the window?

<details><summary>Answer</summary>

It appears at step 8 (`report_openl_context`). The open-loop error starts close to the recon error and generally grows with distance from step 8. Recon stays roughly flat, maybe a bit higher in the first few steps while the latent warms up.
</details>

**M2. Convert MSE to something meaningful.** The metrics box says `recon MSE (forces): 900`. What's the typical error per force component in counts? Is that good?

<details><summary>Answer</summary>

√900 = 30 counts. Compare it with the data. The std of forces is about 100–145 counts and the resting levels are tens of counts, so 30 counts is moderate: fine for hard contacts (hundreds of counts) and noticeable for light ones.
</details>

**M3. Look at an ending.** Take a success episode with length L and set Window start = L − 64. What should the reward and cont curves look like at step 63? Then do the same for a fail episode.

<details><summary>Answer</summary>

- Success: reward gt jumps to +1 and cont gt drops to 0 at step 63.
- Fail: reward gt is −1 at step 63, but cont gt stays 1, because the episode was truncated, not terminated.

Check whether recon and open loop anticipate either one, and in the fail case whether the prediction reacts to the release (6.1).
</details>

**M4. Density and error.** Find a window where the forces' open-loop MSE spikes. Is the "other episodes" neighbour count low at those steps?

<details><summary>Answer</summary>

If it is, the model fails because similar data is rare, and the fix is more (or more varied) demonstrations. If neighbour counts are high and the error is still large, the model itself (capacity, training time, normalisation) is the problem.
</details>

**M5. Ensemble spread.** In the per-prior panel, find a moment where the grey band is wide. Is the ground truth inside the band? What is the worst case for LEQ?

<details><summary>Answer</summary>

A wide band means the heads disagree, which is the uncertainty LEQ is penalised for. The worst case is a **narrow** band that misses the truth: the model is confidently wrong, and the penalty won't protect the policy from it.
</details>

**M6. Train vs val.** After retraining with `val_frac 0.1`, compare open-loop √MSE at step 63 for five train and five val episodes, over the same kinds of windows. What does a large gap mean?

<details><summary>Answer</summary>

A model that does clearly better on train than on val is overfitting: it memorised the training demos. Remedies are more data, a smaller model, regularisation or fewer training steps. Also compare W&B's `report/*` (train batch) with `eval/*` (val batch).
</details>

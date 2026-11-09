# 4. The world model in plain words

`train_model.sh` trains DreamerV3's **world model**: a network that learns *how the hand and its skin behave*. Given what it has felt so far and the next joint command, it predicts what it will feel next. LEQ (the second stage, `train_leq.sh`) later uses it as a simulator, "imagining" what would happen under different actions to learn a policy without touching the real robot.

So the world model must be good at exactly the things the model visualizer shows.

## The parts

```
 observation(t) ──► ENCODER ──► embed(t) ─┐
                                          ▼
 latent(t-1), action(t-1) ──────────► RSSM ──► latent(t) ──► DECODER ──► forces, positions (reconstruction)
                                                        ├─► REWARD head ──► reward
                                                        └─► CONT head   ──► continue? (1 - is_terminal)
```

- **Encoder**: squeezes the 2,208 observation numbers (z-scored and symlogged) into a compact *embedding*.
- **Latent state**: the model's internal summary of "the situation so far". It has two parts:
  - `deter`: a memory, carried forward by a recurrent network (GRU) step by step.
  - `stoch`: a small random part that captures what can't be predicted exactly.
- **RSSM** (Recurrent State-Space Model): updates the latent state at each step. It can do this in two ways:
  - **Posterior** ("observe"): uses the previous latent, the action **and the real observation**. It knows what actually happened.
  - **Prior** ("imagine"): uses only the previous latent and the action. It has to *guess* what happens next. This mode is what LEQ uses to dream.
- **Decoder**: turns a latent state back into forces and positions. Its output is the model's "drawing" of what the skin feels.
- **Reward head** and **continuation head**: predict the reward and whether the episode continues. Continuation is `cont = 1 − is_terminal`, so here it is 1 everywhere except the last step of successful episodes.

## Training in one paragraph

Training repeatedly samples a **window of 65 consecutive steps** (0.65 s) from a random training episode. It runs the posterior over the window and is penalised for three things:

1. Decoder outputs that differ from the real forces and positions (*reconstruction loss*).
2. Reward and continuation predictions that are wrong.
3. Priors that disagree with the posteriors (*dynamics loss*). This teaches the "imagine" mode to guess what the "observe" mode will see.

The model never sees a whole 25-second episode at once. Its memory of the past comes only from the latent state carried along inside the window.

## The two tests the visualizer runs

Both run on one 64-step window that you choose.

### Teacher-forced reconstruction ("recon")

The model gets the **real observation at every step** (posterior), then decodes. It's like asking "describe what you are seeing right now". This tests the encoder and decoder. It should be good, and errors here mean the model can't even represent the data.

### Open-loop imagination ("openl")

The model gets the real observations for the first **8 steps** only (`report_openl_context: 8`). After that it gets **only the actions** and must imagine the remaining 56 steps with the prior. It's like asking "here's what happened for 0.08 s, and here are the commands for the next 0.56 s: what will the skin feel?"

This tests the **dynamics**, and it's what matters for LEQ. Errors grow the further it imagines; that's normal. What matters is how fast they grow and whether the prediction still follows the real trend.

"Teacher forcing" is the machine-learning term for feeding the true previous values instead of the model's own predictions. "Open loop" means no feedback from reality.

## Ensemble priors

The prior has **7 heads** (`dyn.rssm.ensemble: 7`): 7 slightly different guessers. During imagination, each step picks one at random.

- Where the 7 agree, the model is confident.
- Where they disagree, the model is unsure. Usually that's because the situation is rare in the data.

LEQ uses this disagreement as an **uncertainty penalty**, so it won't trust imagined futures in regions where the model is unsure.

[PossibleBug.md](../PossibleBug.md) documents an issue: the 7 heads are chained rather than independent. They probably disagree *less* than true independent heads would, so read the spread as a lower bound on uncertainty.

## Why the window start matters

The visualizer starts every window with **empty memory**: `is_first` is set at the window's first step, whatever came before it in the episode. So:

- the first few steps of recon can be worse while the latent "warms up";
- the model only knows about the past inside the window. If you start a window halfway through a press, the model doesn't know how it began.

Training has the same limitation: it also uses short windows.

## Units in the visualizer

The visualizer converts forces **back to Xela counts** before plotting and before computing MSE. So force errors are in counts² and comparable across runs, whatever the normalisation. Positions are in metres. Position MSEs are therefore tiny numbers like 1e-6 m² (1 mm RMS). See chapter 2.11.

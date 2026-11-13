# Possible bug: ensemble prior heads are chained, not independent

**Status:** fixed behind the flag `dyn.rssm.independent_heads` (default `False`, so old
checkpoints still load). `train_ditto.sh` and `train_leq.sh` copy the flag from the
world model's `config.yaml`. The original LEQ_DV3 implementation has the same bug (it is
already in the first commit, `de1f696 clean up codes`).

**Effect:** see [reports/independent_heads/](reports/independent_heads/README.md).
Retrained at 10, 50 and 100 Hz with 3 seeds each. With the fix, DITTO's command error on
real sensor input falls by 38–60%, and the world model's KL falls slightly. The heads
disagree 9–17% more, but they still underestimate the real error about 5×. Later heads
did not drift more than earlier ones, either before or after the fix.

The note below describes the bug as it was.

## Where

`RSSM._prior` in [`dreamerv3/nets.py`](../../dreamerv3/nets.py):

```python
x = feat                                   # set once, outside the loop
if self.ensemble > 1:
  logits = []
  for j in range(self.ensemble):
    for i in range(self.imglayers):
      x = self.get(f'img{i}_{j}', Linear, self.hidden, **kw)(x)
    logit = self._logit(f'imglogit_{j}', x)
    logits.append(logit)
```

## What it is

The ensemble (`dyn.rssm.ensemble`, 7 for mechanical pen) is meant to be N
independent prior heads, each mapping the GRU feature `feat` to a prior over
the next stochastic state.

`x` is only set to `feat` once, before the loop. So:

- Head 0 takes `feat` as input.
- Head `j > 0` takes the last hidden layer of head `j-1` as input.

The heads form a single chain, which is `imglayers * ensemble` layers deep
(2 * 7 = 14), with a logit read out every `imglayers` layers. Nothing crashes
because every hidden layer has the same width (`hidden -> hidden`). The bug is
silent.

## What it would cause

The priors are used in three places, so the bug reaches all of them:

1. **World-model training** (`RSSM.loss`). The dyn/rep KL is computed
   against every head (`output_all=True`). All heads are trained in this
   chained form, so later heads depend on the weights of every earlier head.
   Their gradients also flow back through the earlier heads.
2. **Imagination** (`RSSM.imagine`). Every imagined step samples one head at
   random. LEQ trains its actor, critic and value on these rollouts, and the
   open-loop plots in `visualize_model.py` use them too.
3. **Uncertainty penalty** (`RSSM.get_penalty`). The penalty is the spread of
   the heads' predicted `z` around the ensemble mean. Because the heads share
   a computation path, they are correlated rather than independent. They will
   probably disagree less than a true ensemble would, so the penalty tends to
   **underestimate model uncertainty**. As a result, LEQ is penalized less for
   exploiting regions where the world model is wrong.

Two smaller effects:

- Later heads are deeper networks than earlier ones, so the heads are not
  identically distributed. This may explain why some heads (e.g. head 4 on
  the mechanical pen checkpoint) drift more in open loop than others.
- Head capacity and training dynamics differ from the design described for an
  ensemble of priors.

## Fix (applied behind `dyn.rssm.independent_heads`)

Reset the input for every head:

```python
for j in range(self.ensemble):
  x = feat
  for i in range(self.imglayers):
    ...
```

This changes the input width of `img0_{j}` for `j > 0` (from `hidden` to the
size of `feat`). **Existing world-model checkpoints will not load** after the
fix, so the world model must be retrained. To keep old checkpoints loading,
put the fix behind a config flag (e.g. `dyn.rssm.independent_heads`) that
defaults to the old behaviour.

## Findings after the fix (2026-11-13)

I retrained the world model and DITTO on the real mechanical pen data at 10, 50 and
100 Hz (`new10hz`, `new50hz`, `new100hz`), with 3 seeds each:

- **Fixed runs:** `logdir/data_sweep_indep`, trained with `--dyn.rssm.independent_heads True`.
- **Baseline:** `logdir/data_sweep`, the chained heads.

Both sweeps used the same data, held-out split, rollout episodes and code. The configs
differ only in the flag. Values below are mean over 3 seeds, old → fixed. **Bold**
means the change is larger than the seed spread (the two standard deviations added).
Full report: [reports/independent_heads/](reports/independent_heads/README.md).

| | 10 Hz | 50 Hz | 100 Hz |
|---|---|---|---|
| Policy command error with real sensor input (rad) | 0.123 → **0.049** | 0.114 → **0.059** | 0.130 → **0.081** |
| DITTO held-out action MSE | 0.0040 → **0.0010** | 0.0053 → **0.0012** | 0.0061 → **0.0037** |
| Policy command error in imagination (rad) | 0.161 → **0.098** | 0.255 → 0.208 | 0.314 → **0.203** |
| Policy jitter with real sensor input (x expert) | 5.52 → **5.08** | 28.8 → **25.7** | 56.4 → 53.9 |
| World-model KL, held-out (nats) | 6.07 → **4.97** | 2.95 → **2.67** | 2.62 → **2.49** |
| Open-loop forces loss, held-out | 8.07 → 7.67 | 5.37 → **4.96** | 5.01 → **4.61** |
| Heads' spread / their error, forces | 0.117 → **0.135** | 0.163 → **0.191** | 0.176 → **0.192** |
| Worst / best head MSE, forces | 1.02 → 1.02 | 1.02 → 1.03 | 1.02 → 1.02 |
| DITTO reward, held-out | 0.956 → 0.952 | 0.986 → 0.987 | 0.991 → 0.991 |

How the findings compare with the predictions above:

1. **World-model training (prediction 1): small gain.** Held-out KL is lower at every
   rate (−5 to −18%), and open-loop forces loss is lower at 50 and 100 Hz. The prior's
   certainty also varies much less between seeds: its seed std drops from about 0.13 to
   about 0.025. Reconstruction and the error when replaying expert actions show no
   change beyond the seed spread.
2. **Imagination (prediction 2): biggest effect, on the policy.** DITTO trains on
   imagined rollouts that pick a random head at each step. With the fix, its commands
   are 38–60% closer to the demonstrator's when it gets real sensor input. Every fixed
   seed beats every old seed at all three rates. A likely reason: the chained heads
   were 2 to 14 layers deep, so the imagined latents changed in character from step
   to step depending on the head drawn. This explanation was not tested directly.
3. **Uncertainty penalty (prediction 3): confirmed, but small.** The heads disagree
   9–17% more relative to their error, so the chained heads did underestimate
   uncertainty. But the spread is still only 0.13–0.19 of the actual open-loop error,
   so the ensemble underestimates its error about 5× either way. For LEQ's penalty,
   the chain was not the main cause.
4. **Later heads drifting more: not supported.** In both versions the worst head's MSE is
   within 1.02–1.03× of the best head's, and the error does not rise with head index.
   The "head 4 drifts more" observation is not a systematic effect of the chain.
5. **Jitter vs rate: largely unchanged.** Jitter improves 8–11% at 10 and 50 Hz with
   real sensor input, and the change at 100 Hz is within the seed spread. The policy is
   still about 5× as jerky as the expert at 10 Hz, 26× at 50 Hz and 54× at 100 Hz. The
   fix does not change which rate to use: 10 Hz is still best.
6. **DITTO reward cannot tell the two apart.** It is close to its ceiling in both
   versions.

**Recommendation:** train every new mechanical pen world model with
`dyn.rssm.independent_heads: True`. Consider making it the default in the
`mechanical_pen:` config block.

**Caveats:** only 3 seeds, and only 3 rolled-out episodes per run, so the
"larger than the seed spread" rule is a rough guide, not a significance test.

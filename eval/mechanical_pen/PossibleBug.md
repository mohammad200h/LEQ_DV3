# Possible bug: ensemble prior heads are chained, not independent

**Status:** not fixed. The original LEQ_DV3 implementation has the same code
(it is already present in the first commit, `de1f696 clean up codes`). This
note only documents it. Nothing in this repo has been changed.

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

## Fix (not applied)

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

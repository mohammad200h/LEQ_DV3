# Input scaling for the mechanical pen data: symlog and normalization

This note covers how DreamerV3 scales its inputs and why that isn't enough for the mechanical pen data on its own. It also explains what we do about it: what is implemented, what isn't yet, and what to watch for.

**Status**

| Key | Scaling applied today |
|---|---|
| `forces_local` | Per-axis z-score (`forces_norm`, enabled in the `mechanical_pen` config), then symlog inside the network |
| `positions` | **Symlog only. Not normalized yet** (see [What is still open](#what-is-still-open)) |
| `action` | Already in `[-1, 1]` (scaled when the chunks were built) |

The numbers below were measured on 21 of the 564 DV3 episodes (every 28th file, 46,920 steps). Rerun the measurement if the data is regenerated.

---

## 1. What symlog is and where DreamerV3 uses it

```
symlog(x) = sign(x) * log(1 + |x|)
symexp(x) = sign(x) * (exp(|x|) - 1)     # inverse
```

`symlog` is roughly the identity near zero (`symlog(0.5) ≈ 0.41`) and logarithmic for large magnitudes (`symlog(1000) ≈ 6.9`). It is symmetric, so negative values work the same way, unlike `log`.

In this repo, symlog is applied in two places for vector inputs such as `forces_local` and `positions`:

- **Encoder input:** `SimpleEncoder` applies `symlog` to every vector key before its MLP ([nets.py:319-320](../../dreamerv3/nets.py#L319-L320), `enc.simple.symlog: True`).
- **Decoder loss:** vector heads use `vecdist: symlog_mse` ([nets.py:635-637](../../dreamerv3/nets.py#L635-L637)). The network predicts in symlog space, the target is passed through `symlog`, and the loss is squared error **summed over every dimension** of the key (`MSEDist`, `agg='sum'`, [jaxutils.py](../../dreamerv3/jaxutils.py)). Predictions are mapped back with `symexp`.

DreamerV3 also uses symexp/two-hot targets for the reward and critic heads, and percentile-based return normalization for the actor. Those are separate from observation scaling and aren't covered here.

## 2. Why DreamerV3 uses symlog instead of normalization

DreamerV3 (Hafner et al., *Mastering Diverse Domains through World Models*, 2023) was built to run **one fixed set of hyperparameters across very different domains**: Atari, DMC, Minecraft, Crafter and others. Their observations and rewards range from fractions of a unit to thousands, and the scale isn't known in advance.

The usual alternative is z-scoring with **running** mean/std statistics. In online RL those statistics drift as the replay buffer fills with new behaviour, so the input the network sees for the same physical state keeps changing. That makes the learning problem non-stationary. Clipping (as in classic Atari reward clipping) is stable but throws information away.

Symlog avoids both problems:

- **No statistics:** it's a fixed function, so nothing drifts.
- **Large values are tamed:** a target of 10,000 becomes about 9.2, so it can't produce huge squared errors or gradients and swamp the loss.
- **Nothing is clipped:** it's invertible, so the full value can be recovered with `symexp`.

The implicit assumption is that signals are roughly **O(1) or larger**. Symlog only ever *shrinks* magnitudes. It never scales a small signal up.

## 3. Why symlog alone does not work for the mechanical pen data

The pen chunks contain two kinds of signal that violate that assumption in opposite directions.

### 3a. Forces are large, so raw symlog turns them into a log scale

`forces_local` holds Xela counts after baseline subtraction and outlier removal. They are **not** scaled when the chunks are written. The processing script only saves the statistics, to `sparsh_stats.json`.

| `forces_local` (raw counts) | x | y | z |
|---|---|---|---|
| mean | 6.8 | 8.3 | 30.1 |
| std | 138.6 | 88.4 | 140.1 |

- Median `|x|` = 22.6, p99 = 577, max = 2862.
- Only **3.3%** of values fall in `|x| < 1`, the region where symlog behaves like the identity.

So for raw counts, symlog is essentially `log(|x|)` everywhere. Three consequences:

1. **Contact magnitude gets flattened.** The max/median ratio drops from **127×** in counts to **2.5×** after symlog. A hard press and a light touch look much more similar to the model than they really are.
2. **Sensitivity becomes relative, not absolute.** Going from 20 to 40 counts and from 200 to 400 counts produce the same change in symlog (about 0.67). A 20-count error on a light contact costs as much as a 200-count error on a hard one.
3. **Errors on hard contacts grow when converted back.** The decoder is trained in symlog space. A small error there on a large value becomes a large error in counts after `symexp`.

### 3b. Positions are tiny, so symlog leaves them tiny

`positions` are taxel coordinates from forward kinematics, in **metres**, in the hand base frame.

| `positions` (m) | x | y | z |
|---|---|---|---|
| mean | 0.017 | -0.058 | 0.101 |
| std | 0.045 | 0.018 | 0.058 |

Near zero, symlog is about the identity, so these stay around 0.05. More importantly, what the model has to *predict* is how positions change over time, and that is very small:

- Median per-dimension temporal variance of `positions` is **2.2e-6**, compared with **3.6** for raw symlog forces.
- RMS change per step is **0.00075** for positions, compared with **0.09** for symlog forces.

### 3c. The two keys share one summed loss

The decoder loss adds up squared error over **all** dimensions of **all** keys. Each key's share of the loss is roughly proportional to its variance in symlog space. With raw data, the position term is about a **millionth** of the force term (median per-dimension variance 2.2e-6 vs 3.6).

So the model can essentially **ignore finger motion** and still report a near-zero `positions` loss. That small number reflects the scale of the data, not a good model of the hand's motion. A small `positions` loss on its own is therefore **not** evidence that the world model tracks the fingers.

The same problem comes back whenever a new key with a different scale is added. Joint angles in radians would be O(1). A 64×64×3 image adds 12,288 summed dimensions and would dominate everything else.

## 4. Why normalize

Normalizing (z-scoring) each channel, `z = (x − μ) / σ`, fixes the parts symlog can't:

- **Scale goes to O(1) in both directions:** large signals are scaled down and small ones scaled **up**, so the keys contribute comparably to the summed loss.
- **It's linear:** a press twice as hard stays twice as far from the mean, so contact magnitude isn't flattened.
- **Symlog still helps afterwards:** it now acts only on the tails, outliers beyond about 1σ.

The usual objection to normalization (statistics that drift during online training) **doesn't apply here**. This is **offline** training on a fixed demo set, so the statistics are computed once and frozen.

## 5. How normalization is applied to this data

### 5a. Forces: implemented

**Config** ([configs.yaml](../../dreamerv3/configs.yaml), `mechanical_pen:` block):

```yaml
forces_norm.enable: True
forces_norm.mean: [6.1331, 8.4556, 28.8512]
forces_norm.std: [140.4322, 94.8270, 145.6174]
```

The global default is `forces_norm: {enable: False, ...}`, so other configs (`leap`, `vd4rl`, ...) are unaffected.

**Where the numbers come from.** `sparsh_processing.py` computes them while writing the episodes and stores them in `ros_ws/demonstrations/compose_playbacks/sparsh_skin/sparsh_stats.json`:

- **Per axis (x, y, z):** one mean/std per axis, pooled over all 368 taxels and all timesteps. The values are **not** computed per taxel. Taxels that are rarely touched have almost no variance, and per-taxel scaling would blow their noise up to std 1.
- **Zeros treated as NaN:** outliers are set to 0 during processing, so zeros are left out of the statistics rather than pulling the mean towards 0.
- **Same statistics as Sparsh:** this is the same normalization Sparsh-skin applies inside its own model, so the two pipelines see consistently scaled inputs.

**Where it runs in the code:**

- `jaxutils.norm_forces` / `jaxutils.denorm_forces` ([jaxutils.py:84-95](../../dreamerv3/jaxutils.py#L84-L95)) compute `(x − mean) / std` and its inverse. They pass the data through unchanged when `enable` is false.
- `preprocess()` in both [model_agent.py](../../dreamerv3/model_agent.py) (world model) and [agent.py](../../dreamerv3/agent.py) (LEQ) applies `norm_forces` to `forces_local`.
- Every training, reporting and policy path goes through `preprocess()`, so **the chunks on disk and any live observations stay in raw counts**. Don't normalize them yourself before feeding the agent, or they'll be normalized twice.
- `model_agent.visualize` converts reconstructions, open-loop predictions and ground truth back to counts with `denorm_forces` ([model_agent.py:419-423](../../dreamerv3/model_agent.py#L419-L423)). The visualizer plots and its MSEs are therefore in Xela counts and comparable across runs, whatever the normalization setting.

**Effect** (same 21 episodes):

| `forces_local` | raw → symlog | z-score → symlog |
|---|---|---|
| std per axis | 3.6 / 3.0 / 3.5 | 0.50 / 0.39 / 0.48 |
| max | 7.96 | 3.19 |
| share in the linear part of symlog (`\|x\| < 1`) | 3.3% | **88.2%** |

After the z-score, the per-axis std is about 1 (1.00 / 0.95 / 0.99) and the mean is about 0. Most of the data now sits where symlog is linear, and only real contact spikes (p99 ≈ 4σ, max ≈ 23σ) get compressed.

### 5b. Positions: not yet normalized

Even with forces normalized, `positions` contribute far less than forces to the summed loss. The total per-dimension variance across all 1104 dimensions is **0.20** for positions against **75** for normalized forces, about **380×** less.

The recommended fix is the same per-axis z-score, using the position statistics above:

```
mean = [0.0169, -0.0580, 0.1014]    std = [0.0453, 0.0179, 0.0582]   (metres)
```

This brings the positions' total variance to about **170**, comparable to the forces (75).

One caveat: the per-axis std of positions mostly measures **where** the taxels sit on the hand (their spatial spread), not how much they move. After the z-score, each taxel still has a static offset, and its motion is a small change on top of that. That is fine for the loss balance (the totals above already account for it), but don't expect the normalized positions to have unit variance over time.

Implementing this would mirror `forces_norm`: a `positions_norm` config entry, a call in both `preprocess()` functions, and a denorm in `visualize`. Per-key loss weights are an alternative if changing the inputs isn't wanted.

## What is still open

1. **Positions are not normalized** (section 5b). Until they are, read the `positions` loss with caution.
2. **The statistics include the validation episodes.** `sparsh_stats.json` is computed over all written episodes, including the 10% held out by `val_frac`. With 564 episodes and per-axis statistics the leak is negligible, but it isn't strictly a clean split.
3. **The statistics are hard-coded in `configs.yaml`.** If the demos are regenerated (new baseline, new episodes), copy the new values from `sparsh_stats.json` into the `mechanical_pen` block. They are not read automatically.
4. **Old checkpoints:** a world model trained before `forces_norm` was enabled expects raw counts. Don't resume it, or train LEQ on top of it, with normalization switched on.

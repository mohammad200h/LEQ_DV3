# 1. Meet the data

## The task

A LEAP hand, covered in XELA tactile sensors, holds a mechanical (click) pen and tries to **press the pen's button with its thumb**. A human teleoperated the hand to record demonstrations. A click detector (audio and tactile) then labelled each recording:

- **success**: the click was heard.
- **fail**: the recording was cut off without a click.

## Where the files come from

```
ros_ws/demonstrations/compose_playbacks/th_1_23.npz         raw recording (video, audio, joint states, Xela)
        └─ sparsh_skin/episode_th_1_23.npz                  resampled to 100 Hz, baseline subtracted, forward kinematics
              └─ DV3/<time>-<uuid>-<uuid>-<steps>.npz       DreamerV3 "chunk": what the model trains on
LEQ_DV3/data/leap/mechanical_pen  ->  .../compose_playbacks/DV3     (symlink)
```

The long chunk file names are what DreamerV3's replay loader expects. `DV3/manifest.json` maps them back to readable names like `th_1_23`. The explorer shows the readable names.

- `th_1_*` episodes: 549
- `th_3_*` episodes: 14
- `th_4_*` episodes: 1

## One episode = a table over time

Each chunk file holds a few arrays. The first dimension is always **time** `T`: one row per step, 100 steps per second. Episodes are 1,176 to 2,840 steps long, so about 12 to 28 seconds.

| key | shape | what it is | units |
|---|---|---|---|
| `forces_local` | `(T, 368, 3)` | force on each of 368 taxels, as an x/y/z vector in the taxel's own frame | Xela counts (raw sensor units, baseline subtracted) |
| `positions` | `(T, 368, 3)` | where each taxel is in space, from the joint angles via forward kinematics | metres, hand base frame |
| `action` | `(T, 16)` | the joint targets sent to the 16 motors | rad scaled to `[-1, 1]` |
| `reward` | `(T,)` | 0 everywhere, except the last step: +1 success, −1 fail | – |
| `is_first` | `(T,)` | True only at step 0 | – |
| `is_last` | `(T,)` | True only at the last step | – |
| `is_terminal` | `(T,)` | True at the last step of **success** episodes only | – |

**How to read a shape.** `(T, 368, 3)` means "for every time step, for every one of 368 taxels, 3 numbers". One episode of 2,000 steps therefore holds 2,000 × 368 × 3 ≈ 2.2 million force numbers. The whole dataset has 1,268,291 steps, which makes it big. That's why the explorer computes statistics once and caches them.

### Taxels, patches and fingers

A **taxel** ("tactile pixel") is one sensing point. The 368 taxels are grouped into 18 **patches**, always in this order:

| finger | patches (taxel count) |
|---|---|
| thumb | tip (30), link15 (16), link14 (16) |
| index | tip (30), link2 (16), link1A (16), link1B (16) |
| middle | tip (30), link6 (16), link5A (16), link5B (16) |
| ring | tip (30), link10 (16), link9A (16), link9B (16) |
| palm | palm 2 (24), palm 1 (24), palm 3 (24) |

That is 4×30 + 11×16 + 3×24 = 368. The order comes from Sparsh's `XELA_FLATTEN_ORDER`, and each episode's metadata records it (`taxel_order`). You can check the mapping is right in the explorer: the "index" taxels sit together at x ≈ +0.05 m, "middle" at x ≈ 0, "ring" at x ≈ −0.04 m and the thumb off to the side.

### Forces: two things to know

1. **The force is a 3D vector.** Plots usually show its **magnitude**, the length of the vector `|F| = sqrt(fx² + fy² + fz²)`, because one number per taxel is easy to colour.
2. **"No contact" does not read 0.** Even after baseline subtraction, many taxels sit at tens of counts when nothing touches them. The median taxel reads 47 counts, and the ring finger averages about 33 counts while it never touches anything. So look for **changes** in force, not for "non-zero".

### Positions

These are not measured by a sensor. They're computed from the joint angles (forward kinematics), so they tell you **where the fingers are**: the hand's shape. They change slowly. The median taxel moves 0.022 mm per step.

### Actions

16 joint targets in this order:

```
if_mcp if_rot if_pip if_dip | mf_mcp mf_rot mf_pip mf_dip | rf_mcp rf_rot rf_pip rf_dip | th_cmc th_axl th_mcp th_ipl
 index finger               | middle finger               | ring finger                 | thumb
```

`mcp`, `pip` and `dip` are the knuckles from palm to tip, `rot` is the sideways spread, and the thumb has its own names. On disk the angle `q` (rad) is scaled as

```
action = 2 * (q - (-1.5)) / (2.5 - (-1.5)) - 1        q in [-1.5, 2.5] rad  ->  action in [-1, 1]
```

so `action = -0.25` means `q = 0 rad`, the hand's open pose. The explorer shows both numbers.

**Timing convention.** `action[t]` is the command sent *after* seeing observation `t`. The converter shifted the commands one step forward to arrange this. The last action of every episode is filled in as 0 rad (`-0.25`), which is why every action line jumps at the very last step.

### Reward and the episode flags

Rewards are **sparse**: 564 out of 564 episodes have exactly one non-zero reward, on the last step.

- Success: reward +1, and `is_terminal=True`. The task really ended.
- Fail: reward −1, and `is_terminal=False`. The episode was *cut off* (truncated) and could, in principle, have continued.

The `README.md` in the parent folder says reward is in `[-1, 0]`. The data actually holds −1, 0 and +1.

## Try it

From `LEQ_DV3/`:

```python
import numpy as np, glob
f = sorted(glob.glob('data/leap/mechanical_pen/*.npz'))[0]
ep = np.load(f)
for k in ep.files:
    print(k, ep[k].shape, ep[k].dtype)
mag = np.linalg.norm(ep['forces_local'], axis=-1)   # (T, 368): |F| per taxel
print('strongest contact at step', mag.max(1).argmax())
print('final reward', ep['reward'][-1], 'terminal?', ep['is_terminal'][-1])
```

Then open the explorer's **Episode** tab, pick `th_1_1` and press **Jump to strongest contact**. It should land on the same step.

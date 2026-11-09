# Glossary

| term | meaning |
|---|---|
| **action** | the 16 joint targets sent to the hand at a step, scaled from [−1.5, 2.5] rad to [−1, 1] |
| **chunk** | DreamerV3's file format for one episode (`<time>-<uuid>-<uuid>-<steps>.npz`) |
| **cont / continuation** | `1 − is_terminal`: will the episode go on after this step? |
| **correlation** | −1…+1, how linearly two quantities move together |
| **coverage / density** | how much data exists near a given state |
| **decoder** | network that turns the latent state back into forces and positions |
| **deter / stoch** | the deterministic (memory) and stochastic (random) parts of the latent state |
| **encoder** | network that turns an observation into an embedding |
| **ensemble / prior heads** | 7 alternative prior networks; their disagreement measures uncertainty |
| **episode** | one recorded demonstration, from start to click (success) or cut-off (fail) |
| **forward kinematics (FK)** | computing where each part of the hand is from the joint angles |
| **histogram** | bar chart of how many values fall into each range (bin) |
| **is_first / is_last / is_terminal** | flags on the first step, the last step, and the last step of a really finished (successful) episode |
| **latent state** | the model's internal compressed summary of the situation |
| **LEQ** | the second training stage: learns a policy inside the frozen world model's imagination |
| **log scale** | axis where each factor of 10 gets the same space |
| **magnitude \|F\|** | length of the 3D force vector, √(fx² + fy² + fz²) |
| **median / percentile** | the value with 50% / p% of the data below it |
| **MSE / RMSE** | mean squared error / its square root (back in normal units) |
| **open loop (openl)** | the model predicts from actions alone, without seeing new observations |
| **overfitting** | doing well on training data and badly on unseen data |
| **patch** | one physical XELA sensor pad (16, 24 or 30 taxels) |
| **PCA / PC1, PC2** | finding the directions of largest variation, to draw high-dimensional data in 2D |
| **posterior / prior** | latent update with / without the real observation |
| **resting level** | what a taxel reads with nothing touching it (tens of counts here) |
| **reward** | +1 at the last step of success, −1 at the last step of fail, 0 otherwise |
| **RSSM** | Recurrent State-Space Model, the world model's dynamics core |
| **std (σ)** | standard deviation: the typical distance from the mean |
| **symlog** | sign(x)·log(1+\|x\|): linear near 0, logarithmic far away |
| **taxel** | one tactile sensing point (368 on this hand) |
| **teacher forcing (recon)** | the model sees the real data at every step |
| **train / val split** | training episodes vs held-out episodes used to check generalisation |
| **truncated** | episode cut off without reaching an end state (all fails here) |
| **violin plot** | a sideways, mirrored histogram with a median marker |
| **window** | a short run of consecutive steps (65 in training, 64 in the visualizer) |
| **world model** | network that learns to predict what the robot will sense next, given its actions |
| **Xela counts** | raw units of the XELA tactile sensors (baseline subtracted) |
| **z-score** | (x − mean) / std: value measured in standard deviations |

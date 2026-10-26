# Logs and observation based on above tools on various settings

## Diagnostic tools

All four tools are cells in [3 deep nn.py](3%20deep%20nn.py). Tools 1 to 3 read the state of the **last** training step, so they describe the network after the run, not at init.

### Tool 1: Forward-pass diagnostic, activation distribution per Tanh layer

**What it shows:** a histogram of each Tanh layer's output (`layer.out`) from the last forward pass, plus mean, std and % saturation (`|out| > 0.97`).

**How to read it:**

- Healthy: the std is roughly constant across layers, the histograms have a similar shape, and saturation is low (a few %).
- Shrinking std, or mass piling at 0: the activations are vanishing (init gain too small).
- Mass piled at -1 and +1, or high saturation: tanh is saturated (init gain too large). Gradients through saturated units are about 0, so those neurons stop learning.
- A std near 1 is not "healthy" for tanh. It is the maximum, and it means saturation. Always read the std together with the saturation %.

**What changes it:** the weight init scale (gain) and the presence of BatchNorm.

**Requires:** each Tanh layer stores its output (`self.out = ...`) in the forward pass.

### Tool 2: Backward-pass diagnostic, gradient distribution per Tanh layer

**What it shows:** a histogram of the gradient flowing into each Tanh layer's output (`layer.out.grad`) from the last backward pass, plus mean and std.

**How to read it:**

- Healthy: the gradient std is similar across layers, so every layer learns at a comparable rate.
- Std shrinking toward the early layers: vanishing gradients.
- Std growing toward the early layers: exploding gradients.
- Compare with Tool 1: saturated activations are the usual cause of vanishing gradients.
- The absolute values are small because the loss is averaged over the batch. Compare layers with each other, not against a fixed number.

**Requires:** `layer.out.retain_grad()` called in the forward pass, and the tool run after `loss.backward()` (otherwise `.grad` is `None`).

### Tool 3: Backward-pass diagnostic, weight gradient distribution per Linear layer

**What it shows:** a histogram of `W.grad` for each Linear layer (biases skipped), plus a table of grad mean, parameter std and the grad/param ratio (std of `W.grad` divided by std of `W`).

> In the **Part A** logs the second column is the **weight** std, not the grad std (the grad std is the ratio times that column). From **Part B** on the code prints the real grad std, so the weight std is the column divided by the ratio. For example, B1 layer 3: 0.00124 / 0.00125 = weight std 0.99.

**How to read it:**

- The ratio is the key number. It approximates the relative size of one update step, `lr * ratio`. A healthy update is roughly 1e-3 (the exact scale depends on `lr`).
- A ratio far below that for a layer: that layer barely learns.
- A ratio far above that: its updates are large and may destabilise training.
- Layers should have similar ratios and similar histogram widths. A layer that is an outlier (for example the last layer, when its output is scaled down at init) is where the imbalance is.
- The mean should sit near 0 for every layer.

**Difference from Tool 2:** Tool 2 looks at gradients with respect to activations (the signal flowing through the network). This one looks at gradients with respect to the weights, which is what the optimizer uses to update.

**Requires:** run after `loss.backward()` so `W.grad` is populated, and `layer.parameters()[0]` must be the weight matrix.

### Tool 4: Training-time diagnostic, update-to-parameter ratio over time

**What it shows:** for every weight matrix, the log10 of (std of the update `lr * grad`) / (std of the parameter), recorded at every training step in `ud_tracking`. There is one line per parameter, the x-axis is the step, and the horizontal line marks the -3 reference (1e-3).

**How to read it:**

- Rule of thumb: each update should change a weight by about 1/1000 of its scale, so the lines should hover around -3.
- Lines well below -3: learning too slowly (`lr` too low, or that layer's gradients are tiny).
- Lines well above -3: updates are too large, with a risk of instability.
- Lines should be roughly parallel and stable. A layer drifting away from the rest is under- or over-trained relative to the others.
- The early steps are noisy while the network settles, so judge the trend.
- An `lr` drop (for example 0.1 to 0.01) shows as a downward step of exactly 1.

**Difference from Tool 3:** Tool 3 is a single snapshot of grad/param at the last step. Tool 4 tracks the actual update (including `lr`) across the whole run.

**Requires:** `ud_tracking` filled inside the training loop, one list of log10 ratios per step, in the same order as `params`. Biases and gains (any param with a dimension of size 1) are skipped by the plotting condition.

## Experiment plan

### Terms

| Term | Meaning |
|---|---|
| weight gain | A constant multiplied onto the Linear weights after init. The usual value for tanh is 5/3. It offsets the shrinking effect of tanh so the activation std stays stable across layers. |
| bngain | BatchNorm's learned scale parameter. It is a different thing from weight gain. |
| fan_in | The number of inputs to a Linear layer (`w.shape[0]`). |

### Knobs (all in the baseline cell)

| Knob | What to change |
|---|---|
| INIT | The scale of `Linear.w`. Either the `/ in_features ** 0.5` on line 67 (fan-in scaling), or the weight gain in `layer.w *= X` (X = 1, 0.1, 5/3, ...). |
| BN | Keep or remove the `BatchNorm1D` layers. If you remove them, set `bias=True` on the hidden Linear layers. |
| LR | The `lr = ...` line in the training loop. |
| LAST | The scale of the last layer: `layers[-1].bngain *= 0.1` with BN, or `layers[-1].w *= 0.1` without BN. |

**Keep fixed:** `MANUAL_SEED`, `emb_size`, `n_hidden`, `batch_size` and the number of steps.

The loop breaks at epoch 1000, which is enough for the diagnostics. For the final train/dev loss, run more steps. After each run, look at Tools 1, 2, 3 and 4, then at the final train/dev loss.

Do the runs in this order and change only **one** knob at a time.

### Part A: Init without BatchNorm

Shows why init matters.

| Run | Setting | Expect |
|---|---|---|
| A1 | `w = randn` (std 1), no BN, lr 0.1 | tanh saturates in every layer. |
| A2 | `w = randn * 0.1`, no BN, lr 0.1 | Init is too small, so the activations shrink toward 0 with depth. |
| A3 | `w = randn / sqrt(fan_in)`, no BN, weight gain 1, lr 0.1 | Activations still shrink slowly with depth, because tanh squashes them. |
| A4 | `w = randn / sqrt(fan_in)`, weight gain 5/3, no BN, lr 0.1 | The activation std stays stable across layers. |
| A5 | `w = randn / sqrt(fan_in)`, weight gain 3, no BN, lr 0.1 | The gain is too high, so tanh saturates. |
| A6 | Same as A4, but with the last layer's `w` multiplied by 0.1 | The model starts with low confidence, so the initial loss is close to ln(27) = 3.30. |

### Part B: Init with BatchNorm

Shows BN making init matter less.

| Run | Setting | Expect |
|---|---|---|
| B1 | `w = randn` (std 1), BN, lr 0.1 | Tool 1 looks fine because BN renormalizes. Check Tools 3 and 4. |
| B2 | `w = randn * 0.1`, BN, lr 0.1 | The activations look fine, but the grad/param ratio changes, because BN removes the weight scale from the forward pass. |
| B3 | `w = randn / sqrt(fan_in)`, BN, lr 0.1 | The reference "good" run. |
| B4 | `w = randn / sqrt(fan_in)`, weight gain 5/3, BN, lr 0.1 | |
| B5 | Same as B3, but with the last layer's `bngain` multiplied by 0.1 | Low-confidence start. |

Compare B1, B2 and B3 on Tool 4: the update ratio differs with the weight scale.

### Part C: Learning rate

Use B3 as the base.

| Run | Setting | Expect |
|---|---|---|
| C1 | lr 0.001 | The ud ratio is far below -3 and the loss barely moves. |
| C2 | lr 0.01 | |
| C3 | lr 0.1 | Same as B3. |
| C4 | lr 1.0 | The ud ratio is far above -3, and training is unstable (loss spikes). |

Repeat C1 and C4 on A3 (no BN) to see whether BN tolerates a larger lr.

### Part D: BatchNorm details

Use B3 as the base.

| Run | Setting | Expect |
|---|---|---|
| D1 | BN only in some layers (for example, remove it from the first 2 hidden layers) | |
| D2 | BN momentum 0.1 vs 0.001 | Train loss should be the same. Compare the dev/test loss, which uses the running statistics. |
| D3 | `batch_size` 4 vs 32 vs 256 | Smaller batches give noisier BN statistics. |

### Part E: Depth

Use A4 and B3 as the base.

| Run | Setting | Question |
|---|---|---|
| E1 | 3 vs 5 vs 10 hidden layers | Does the activation std stay stable? Without BN, does the gradient std vanish? |

### What to note for each run

| Source | Look at | Question |
|---|---|---|
| Tool 1 | std per layer and % saturation | Is it roughly constant across layers? |
| Tool 2 | grad std per layer | Same in all layers, or vanishing/exploding? |
| Tool 3 | grad/param ratio per layer | Is any layer far from the others? |
| Tool 4 | ud curves vs the -3 line | Do they settle near -3 or far from it? |
| Loss | first-step loss (expect about 3.3), final train/dev loss | |

### Log template

```text
[A1] setting: ...
     Tool 1: ...
     Tool 2: ...
     Tool 3: ...
     Tool 4: ...
     loss  : initial=..., train=..., dev=...
     takeaway: ...
```

---

# Part A

## A1: w = randn (std 1), no BN, lr 0.1

Total Params: 45727
Initial Loss: 19.2619
Losses: train_loss=tensor(3.9839), dev_loss=tensor(4.5665) and test_loss=tensor(4.4950)


### Tool 1

Layer  1/Tanh >> mean: 0.02014, std: 0.97598 with 89.72% saturation
Layer  3/Tanh >> mean: -0.02873, std: 0.98379 with 93.09% saturation
Layer  5/Tanh >> mean: -0.00554, std: 0.97974 with 90.97% saturation
Layer  7/Tanh >> mean: -0.04856, std: 0.97726 with 91.28% saturation
Layer  9/Tanh >> mean: -0.00580, std: 0.98320 with 93.03% saturation

### Tool 2

Layer  1/Tanh >> mean: -0.00588, std: 0.20503
Layer  3/Tanh >> mean: 0.00173, std: 0.08735
Layer  5/Tanh >> mean: 0.00022, std: 0.04524
Layer  7/Tanh >> mean: 0.00028, std: 0.02453
Layer  9/Tanh >> mean: 0.00036, std: 0.01871

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	    -0.00595	1.99794		0.12919
 2	    +0.00064	1.22586		0.07433
 4	    +0.00046	1.07763		0.04123
 6	    -0.00031	1.01426		0.02423
 8	    -0.00004	1.01599		0.01286
10	    +0.00000	0.73977		0.04740

### Takeaway

Plain `randn` init (std 1) with no normalisation saturates every tanh layer, and training does not recover from it in 1000 steps.

**Forward pass (Tool 1)**
- 90 to 93% of the activations sit at ±1 in every layer, and the std is about 0.98 in all of them, with means near 0.
- A std near 1 looks stable, but for tanh it is the maximum, so it means saturation. The std only makes sense read together with the saturation %.
- The cause: with fan_in 100 and weight std 1, each pre-activation has std of about 10, far outside the range where tanh is linear. The layers do not get worse with depth because they are already at the ceiling.

**Initial loss**
- 19.26 against about 3.30 (ln 27) for a uniform guess. The last layer has std-1 weights on saturated ±1 inputs, so the logits are huge and the model starts out very confident and wrong.
- After 1000 steps the loss is train 3.98, dev 4.57 and test 4.50, which is still worse than a uniform guess. The first steps go on shrinking the logits, and the last layer's weight std fell from 1 to 0.74.

**Backward pass (Tool 2)**
- The gradient std grows toward the input: 0.0187, 0.0245, 0.0452, 0.0874 and 0.205 for layers 9, 7, 5, 3 and 1. That is 11x from the last tanh layer to the first, so the gradients grow rather than vanish.
- A plausible reason: std-1 weights amplify the gradient by about sqrt(100) per layer, while the saturated tanh slope (near 0) shrinks it. The two do not cancel.

**Weight gradients (Tool 3)**
- The grad/param ratio runs from 0.129 (layer 0) down to 0.013 (layer 8), a 10x spread. With `lr = 0.1`, the first layer's update is about 1.3% of its weight scale per step (log10 -1.9), and layer 8's is about 0.13% (-2.9).
- The layers learn at very different rates. The first layer also moved the most: its weight std went from 1 to 2.0, while layers 2 to 8 stayed at 1.0 to 1.2.
- The second column of the table is the weight std, not the grad std. The grad std is ratio x column: about 0.258, 0.091, 0.044, 0.025, 0.013 and 0.035.

**Lesson:** std-1 init gives an overconfident start, saturated tanh everywhere, and updates that are far too uneven across layers. The init has to scale with `1/sqrt(fan_in)` (and tanh's 5/3 gain) so the pre-activations stay in the linear part of tanh (A3 and A4).


## A2: `w = randn * 0.1`, no BN, lr 0.1

Total Params: 45727
Initial Loss: 6.7061

### Tool 1

Layer  1/Tanh >> mean: 0.00863, std: 0.44452 with 0.31% saturation
Layer  3/Tanh >> mean: -0.00962, std: 0.35186 with 0.75% saturation
Layer  5/Tanh >> mean: -0.00591, std: 0.24790 with 0.00% saturation
Layer  7/Tanh >> mean: -0.01475, std: 0.17584 with 0.00% saturation
Layer  9/Tanh >> mean: -0.00479, std: 0.48042 with 0.00% saturation

### Tool 2

Layer  1/Tanh >> mean: 0.00001, std: 0.00457
Layer  3/Tanh >> mean: -0.00009, std: 0.00536
Layer  5/Tanh >> mean: 0.00002, std: 0.00635
Layer  7/Tanh >> mean: -0.00007, std: 0.00860
Layer  9/Tanh >> mean: -0.00066, std: 0.02778

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00035	0.11447		0.18552
 2	-0.00021	0.09398		0.11911
 4	+0.00010	0.08728		0.13746
 6	+0.00007	0.08020		0.15464
 8	+0.00058	0.11344		0.20029
10	-0.00000	0.97129		0.02357

### Takeaway

1. Not much saturation of the tanh activations
2. out.grad still shrinks as it goes deep (from backwards)
3. grad is widening
4. The grad/param ratio varies significantly across layers, indicating uneven learning rates.

## A3: `w = randn * 1/sqrt(fan_in)`, no BN, lr 0.1

Total Params: 45727
Initial Loss: 3.7412

train_loss=tensor(2.2443), dev_loss=tensor(2.5628) and test_loss=tensor(2.5604)

### Tool 1

Layer  1/Tanh >> mean: 0.00212, std: 0.63586 with 3.59% saturation
Layer  3/Tanh >> mean: -0.03237, std: 0.55684 with 0.88% saturation
Layer  5/Tanh >> mean: -0.03470, std: 0.55703 with 1.03% saturation
Layer  7/Tanh >> mean: -0.02504, std: 0.56542 with 1.84% saturation
Layer  9/Tanh >> mean: -0.01581, std: 0.57807 with 1.28% saturation

### Tool 2

Layer  1/Tanh >> mean: 0.00001, std: 0.00301
Layer  3/Tanh >> mean: 0.00001, std: 0.00278
Layer  5/Tanh >> mean: 0.00001, std: 0.00270
Layer  7/Tanh >> mean: -0.00005, std: 0.00287
Layer  9/Tanh >> mean: -0.00006, std: 0.00320

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00008	0.19003		0.05933
 2	+0.00004	0.10516		0.07103
 4	-0.00009	0.10482		0.05979
 6	+0.00000	0.10377		0.06245
 8	+0.00003	0.10513		0.06803
10	+0.00000	0.11172		0.16830

### Takeaway

1. Tanh activations have little saturation, but ok.
2. out.grad is in similar mean and std across layers, indicating stable gradient flow.
3. Still some diff in the param.grad's distribution and inconsistent grad/param ratio across layers. Last layer is updating relatively faster than the others.
4. rate of param update is more or less consistent.

## A4: `w = randn / sqrt(fan_in)`, weight gain 5/3, no BN, lr 0.1

Total Params: 45727
Initial Loss: 4.1738
train_loss=tensor(2.2057), dev_loss=tensor(2.5649) and test_loss=tensor(2.5718)

### Tool 1

Layer  1/Tanh >> mean: 0.00332, std: 0.76485 with 20.59% saturation
Layer  3/Tanh >> mean: -0.03342, std: 0.72655 with 13.84% saturation
Layer  5/Tanh >> mean: -0.03983, std: 0.74617 with 15.38% saturation
Layer  7/Tanh >> mean: 0.00993, std: 0.75154 with 18.22% saturation
Layer  9/Tanh >> mean: -0.02829, std: 0.76432 with 17.75% saturation

### Tool 2

Layer  1/Tanh >> mean: -0.00007, std: 0.00388
Layer  3/Tanh >> mean: 0.00004, std: 0.00372
Layer  5/Tanh >> mean: 0.00006, std: 0.00366
Layer  7/Tanh >> mean: -0.00009, std: 0.00393
Layer  9/Tanh >> mean: -0.00005, std: 0.00429


### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00014	0.30690		0.03879
 2	+0.00005	0.16935		0.05541
 4	+0.00010	0.16967		0.05144
 6	+0.00008	0.16820		0.05396
 8	+0.00001	0.17045		0.05425
10	+0.00000	0.15925		0.15039

### Takeaway

1. Tanh activations have more saturation compared to previous runs, but consistent std.
2,3,4 are same as previous runs.

## A5: `w = randn / sqrt(fan_in)`, weight gain 3, with BN, lr 0.1

Total Params: 45727
Initial Loss: 5.6588

### Tool 1

Layer  1/Tanh >> mean: -0.00732, std: 0.87361 with 50.72% saturation
Layer  3/Tanh >> mean: -0.05081, std: 0.87310 with 51.97% saturation
Layer  5/Tanh >> mean: -0.04597, std: 0.88385 with 53.84% saturation
Layer  7/Tanh >> mean: -0.00126, std: 0.88987 with 54.78% saturation
Layer  9/Tanh >> mean: -0.00881, std: 0.88765 with 55.03% saturation

### Tool 2

Layer  1/Tanh >> mean: 0.00007, std: 0.00839
Layer  3/Tanh >> mean: 0.00011, std: 0.00687
Layer  5/Tanh >> mean: 0.00008, std: 0.00583
Layer  7/Tanh >> mean: 0.00013, std: 0.00536
Layer  9/Tanh >> mean: 0.00003, std: 0.00552

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00075	0.55558		0.03386
 2	+0.00019	0.30512		0.04301
 4	+0.00002	0.30360		0.03747
 6	-0.00005	0.29980		0.03381
 8	+0.00008	0.30228		0.03106
10	+0.00000	0.23795		0.11495

### Takeaway

1. Tanh activations have high saturation, indicating potential vanishing gradient issues.


## A6: same as A4, but with the last layer's w multiplied by 0.1

Total Params: 45727
Initial Loss: 3.8477

train_loss=tensor(2.1999), dev_loss=tensor(2.5500) and test_loss=tensor(2.5527)


### Tool 1

Layer  1/Tanh >> mean: 0.00027, std: 0.76338 with 19.97% saturation
Layer  3/Tanh >> mean: -0.02253, std: 0.72360 with 13.03% saturation
Layer  5/Tanh >> mean: -0.04321, std: 0.73795 with 14.22% saturation
Layer  7/Tanh >> mean: -0.01415, std: 0.74861 with 16.97% saturation
Layer  9/Tanh >> mean: -0.01656, std: 0.74356 with 15.62% saturation

### Tool 2

Layer  1/Tanh >> mean: -0.00002, std: 0.00384
Layer  3/Tanh >> mean: 0.00009, std: 0.00353
Layer  5/Tanh >> mean: 0.00004, std: 0.00337
Layer  7/Tanh >> mean: -0.00014, std: 0.00346
Layer  9/Tanh >> mean: -0.00002, std: 0.00327

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00010	0.30627		0.03713
 2	-0.00003	0.16882		0.05319
 4	-0.00017	0.16948		0.04639
 6	+0.00011	0.16841		0.04890
 8	-0.00003	0.17093		0.04595
10	+0.00000	0.11244		0.21371

Seems bad for me, as last layer's param update is much faster compared to the other layers, which could lead to instability in training.

### Takeaway

1. Reducing the last layer's weight by 0.1 decreases its gradient magnitude relative to other layers, but it still updates faster, indicating potential instability.
2. Tanh activations are less saturated compared to A5, suggesting better gradient flow in earlier layers.

# Part B

## B1: `w = randn` (std 1), BN, lr 0.1

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.5774), dev_loss=tensor(2.8096) and test_loss=tensor(2.8086)


### Tool 1

Layer  2/Tanh >> mean: -0.00762, std: 0.63653 with 2.47% saturation
Layer  5/Tanh >> mean: 0.00146, std: 0.63436 with 2.66% saturation
Layer  8/Tanh >> mean: 0.00206, std: 0.63489 with 2.44% saturation
Layer 11/Tanh >> mean: -0.00903, std: 0.63254 with 2.47% saturation
Layer 14/Tanh >> mean: 0.00022, std: 0.63326 with 2.50% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00345
Layer  5/Tanh >> mean: 0.00000, std: 0.00318
Layer  8/Tanh >> mean: -0.00000, std: 0.00286
Layer 11/Tanh >> mean: -0.00000, std: 0.00249
Layer 14/Tanh >> mean: 0.00000, std: 0.00219

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00006	0.00243		0.00243
 3	+0.00000	0.00124		0.00125
 6	+0.00002	0.00109		0.00109
 9	-0.00001	0.00092		0.00092
12	+0.00000	0.00082		0.00081
15	+0.00002	0.00156		0.00157


### Takeaways

- No saturation in the output activations from tanh
- the grad going through is fairly ok, but the std() is expanding slighly as we go deep from last layer to first layer
- The distribution of W.grad is bad std shrinks as we go deeper, indicating vanishing gradients

## B2: `w = randn * 0.1`, BN, lr 0.1

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.2877), dev_loss=tensor(2.5940) and test_loss=tensor(2.5930)


### Tool 1

Layer  2/Tanh >> mean: -0.00057, std: 0.64056 with 2.69% saturation
Layer  5/Tanh >> mean: -0.00278, std: 0.64350 with 3.00% saturation
Layer  8/Tanh >> mean: -0.00354, std: 0.64532 with 2.41% saturation
Layer 11/Tanh >> mean: -0.00484, std: 0.64911 with 2.09% saturation
Layer 14/Tanh >> mean: 0.00236, std: 0.65121 with 1.69% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00252
Layer  5/Tanh >> mean: -0.00000, std: 0.00239
Layer  8/Tanh >> mean: 0.00000, std: 0.00224
Layer 11/Tanh >> mean: -0.00000, std: 0.00210
Layer 14/Tanh >> mean: -0.00000, std: 0.00209


### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00028	0.01625		0.14036
 3	-0.00005	0.00734		0.07138
 6	+0.00002	0.00660		0.06414
 9	+0.00002	0.00581		0.05691
12	+0.00002	0.00566		0.05452
15	+0.00025	0.01204		0.11038

### Takeaways

- compared to before grad/param ratio is more and we are training at higher speed.

## B3: `w = randn / sqrt(fan_in)`, BN, lr 0.1

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.2867), dev_loss=tensor(2.5981) and test_loss=tensor(2.5986)

### Tool 1

Layer  2/Tanh >> mean: -0.00558, std: 0.63759 with 2.41% saturation
Layer  5/Tanh >> mean: -0.00528, std: 0.64195 with 2.84% saturation
Layer  8/Tanh >> mean: -0.00283, std: 0.64306 with 2.22% saturation
Layer 11/Tanh >> mean: -0.00815, std: 0.64488 with 2.31% saturation
Layer 14/Tanh >> mean: -0.00627, std: 0.64767 with 2.09% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00277
Layer  5/Tanh >> mean: -0.00000, std: 0.00238
Layer  8/Tanh >> mean: -0.00000, std: 0.00229
Layer 11/Tanh >> mean: 0.00000, std: 0.00217
Layer 14/Tanh >> mean: -0.00000, std: 0.00212

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00000	0.01101		0.05893
 3	-0.00004	0.00742		0.07159
 6	-0.00000	0.00604		0.05858
 9	+0.00001	0.00577		0.05647
12	+0.00011	0.00570		0.05491
15	+0.00017	0.01250		0.11459

### Takeaways

- Tool 1 and 2 same
- First layer grad/param outlier is removed

## B4:  `w = randn / sqrt(fan_in)`, weight gain 5/3, BN, lr 0.1

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.2934), dev_loss=tensor(2.6296) and test_loss=tensor(2.6290)


### Tool 1

Layer  2/Tanh >> mean: -0.00751, std: 0.63530 with 2.50% saturation
Layer  5/Tanh >> mean: -0.00390, std: 0.64164 with 2.84% saturation
Layer  8/Tanh >> mean: -0.00119, std: 0.64156 with 2.00% saturation
Layer 11/Tanh >> mean: -0.00760, std: 0.64534 with 2.06% saturation
Layer 14/Tanh >> mean: -0.00042, std: 0.64648 with 2.06% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00359
Layer  5/Tanh >> mean: -0.00000, std: 0.00323
Layer  8/Tanh >> mean: -0.00000, std: 0.00302
Layer 11/Tanh >> mean: -0.00000, std: 0.00293
Layer 14/Tanh >> mean: -0.00000, std: 0.00273

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00000	0.00871		0.02850
 3	-0.00008	0.00697		0.04159
 6	-0.00000	0.00592		0.03516
 9	-0.00000	0.00585		0.03496
12	+0.00003	0.00526		0.03088
15	+0.00009	0.01066		0.06264

### Takeaways

- grad/param ratio decreases because of the weight gain 5/3 initialization

## B5:  Same as B3, but with the last layer's `bngain` multiplied by 0.1

Total Params: 46781
Initial Loss: 3.2892
train_loss=tensor(2.3110), dev_loss=tensor(2.6335) and test_loss=tensor(2.6274)

### Tool 1

Layer  2/Tanh >> mean: -0.00779, std: 0.63553 with 2.69% saturation
Layer  5/Tanh >> mean: 0.00117, std: 0.64184 with 2.72% saturation
Layer  8/Tanh >> mean: -0.00460, std: 0.64156 with 2.38% saturation
Layer 11/Tanh >> mean: 0.00046, std: 0.64647 with 2.00% saturation
Layer 14/Tanh >> mean: 0.00189, std: 0.64713 with 1.84% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00294
Layer  5/Tanh >> mean: 0.00000, std: 0.00246
Layer  8/Tanh >> mean: -0.00000, std: 0.00235
Layer 11/Tanh >> mean: -0.00000, std: 0.00221
Layer 14/Tanh >> mean: -0.00000, std: 0.00196

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00020	0.01169		0.06294
 3	-0.00015	0.00832		0.08111
 6	-0.00004	0.00703		0.06865
 9	-0.00003	0.00641		0.06318
12	+0.00004	0.00572		0.05552
15	+0.00010	0.01067		0.10141

### Takeaways

- only diff is better initial loss start

# Part C

## C1: lr 0.001

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(3.2336), dev_loss=tensor(3.3211) and test_loss=tensor(3.3189)


### Tool 1

Layer  2/Tanh >> mean: -0.00357, std: 0.63864 with 2.28% saturation
Layer  5/Tanh >> mean: -0.00053, std: 0.63616 with 2.72% saturation
Layer  8/Tanh >> mean: -0.00336, std: 0.63674 with 2.34% saturation
Layer 11/Tanh >> mean: -0.00379, std: 0.63659 with 2.59% saturation
Layer 14/Tanh >> mean: -0.00106, std: 0.63679 with 2.09% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00605
Layer  5/Tanh >> mean: 0.00000, std: 0.00550
Layer  8/Tanh >> mean: -0.00000, std: 0.00503
Layer 11/Tanh >> mean: -0.00000, std: 0.00453
Layer 14/Tanh >> mean: 0.00000, std: 0.00395

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00066	0.02442		0.13390
 3	-0.00002	0.02214		0.22275
 6	+0.00025	0.01861		0.18595
 9	-0.00009	0.01667		0.16739
12	+0.00002	0.01524		0.14994
15	-0.00048	0.02736		0.27460


### Takeaways

- loss barely moves

## C2: lr 0.01

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.6972), dev_loss=tensor(2.8670) and test_loss=tensor(2.8716)


### Tool 1

Layer  2/Tanh >> mean: -0.00466, std: 0.63754 with 2.31% saturation
Layer  5/Tanh >> mean: -0.00003, std: 0.64040 with 2.66% saturation
Layer  8/Tanh >> mean: -0.00057, std: 0.63510 with 2.06% saturation
Layer 11/Tanh >> mean: -0.00366, std: 0.64192 with 2.25% saturation
Layer 14/Tanh >> mean: 0.00263, std: 0.64408 with 1.88% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00535
Layer  5/Tanh >> mean: 0.00000, std: 0.00459
Layer  8/Tanh >> mean: -0.00000, std: 0.00429
Layer 11/Tanh >> mean: 0.00000, std: 0.00392
Layer 14/Tanh >> mean: 0.00000, std: 0.00362

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00013	0.02175		0.11913
 3	-0.00035	0.01886		0.18939
 6	-0.00005	0.01566		0.15631
 9	-0.00013	0.01481		0.14855
12	+0.00011	0.01265		0.12436
15	-0.00002	0.02558		0.25589


## C3: lr 0.1

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.2867), dev_loss=tensor(2.5981) and test_loss=tensor(2.5986)

### Tool 1

Layer  2/Tanh >> mean: -0.00558, std: 0.63759 with 2.41% saturation
Layer  5/Tanh >> mean: -0.00528, std: 0.64195 with 2.84% saturation
Layer  8/Tanh >> mean: -0.00283, std: 0.64306 with 2.22% saturation
Layer 11/Tanh >> mean: -0.00815, std: 0.64488 with 2.31% saturation
Layer 14/Tanh >> mean: -0.00627, std: 0.64767 with 2.09% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00277
Layer  5/Tanh >> mean: -0.00000, std: 0.00238
Layer  8/Tanh >> mean: -0.00000, std: 0.00229
Layer 11/Tanh >> mean: 0.00000, std: 0.00217
Layer 14/Tanh >> mean: -0.00000, std: 0.00212

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00000	0.01101		0.05893
 3	-0.00004	0.00742		0.07159
 6	-0.00000	0.00604		0.05858
 9	+0.00001	0.00577		0.05647
12	+0.00011	0.00570		0.05491
15	+0.00017	0.01250		0.11459

## C4: lr 1.0

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.3739), dev_loss=tensor(2.7713) and test_loss=tensor(2.7636)


### Tool 1

Layer  2/Tanh >> mean: -0.00582, std: 0.66628 with 4.81% saturation
Layer  5/Tanh >> mean: 0.00273, std: 0.66820 with 4.50% saturation
Layer  8/Tanh >> mean: -0.01209, std: 0.66842 with 3.09% saturation
Layer 11/Tanh >> mean: 0.00521, std: 0.67081 with 1.66% saturation
Layer 14/Tanh >> mean: -0.00062, std: 0.67310 with 1.22% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00168
Layer  5/Tanh >> mean: 0.00000, std: 0.00140
Layer  8/Tanh >> mean: 0.00000, std: 0.00125
Layer 11/Tanh >> mean: 0.00000, std: 0.00117
Layer 14/Tanh >> mean: -0.00000, std: 0.00124


### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00009	0.00472		0.01877
 3	-0.00001	0.00229		0.01739
 6	-0.00003	0.00183		0.01483
 9	+0.00001	0.00173		0.01435
12	-0.00001	0.00167		0.01386
15	+0.00000	0.00347		0.01988

# Part D

## D1: BN only in some layers (for example, remove it from the first 2 hidden layers)

Total Params: 46181
Initial Loss: 3.6571
train_loss=tensor(2.2821), dev_loss=tensor(2.5974) and test_loss=tensor(2.5900)


### Tool 1

Layer  1/Tanh >> mean: -0.00821, std: 0.63205 with 3.84% saturation
Layer  3/Tanh >> mean: -0.03995, std: 0.55074 with 0.91% saturation
Layer  5/Tanh >> mean: -0.03169, std: 0.52802 with 0.34% saturation
Layer  8/Tanh >> mean: -0.00922, std: 0.64778 with 2.47% saturation
Layer 11/Tanh >> mean: -0.00224, std: 0.65215 with 1.88% saturation

### Tool 2

Layer  1/Tanh >> mean: 0.00001, std: 0.00299
Layer  3/Tanh >> mean: 0.00002, std: 0.00302
Layer  5/Tanh >> mean: -0.00000, std: 0.00314
Layer  8/Tanh >> mean: -0.00000, std: 0.00248
Layer 11/Tanh >> mean: 0.00000, std: 0.00238

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00021	0.01086		0.05748
 2	-0.00007	0.00775		0.07403
 4	+0.00007	0.00632		0.06098
 6	-0.00013	0.00664		0.06506
 9	-0.00001	0.00643		0.06186
12	+0.00017	0.01278		0.11679

## D2: BN momentum 0.1 vs 0.001

0.1 is already B3. 0.001 will be compared against it.

Total Params: 46781
Initial Loss: 3.6346
train_loss=tensor(2.2445), dev_loss=tensor(2.5393) and test_loss=tensor(2.5421)


### Tool 1

Layer  2/Tanh >> mean: -0.00558, std: 0.63759 with 2.41% saturation
Layer  5/Tanh >> mean: -0.00528, std: 0.64195 with 2.84% saturation
Layer  8/Tanh >> mean: -0.00283, std: 0.64306 with 2.22% saturation
Layer 11/Tanh >> mean: -0.00815, std: 0.64488 with 2.31% saturation
Layer 14/Tanh >> mean: -0.00627, std: 0.64767 with 2.09% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00277
Layer  5/Tanh >> mean: -0.00000, std: 0.00238
Layer  8/Tanh >> mean: -0.00000, std: 0.00229
Layer 11/Tanh >> mean: 0.00000, std: 0.00217
Layer 14/Tanh >> mean: -0.00000, std: 0.00212

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00000	0.01101		0.05893
 3	-0.00004	0.00742		0.07159
 6	-0.00000	0.00604		0.05858
 9	+0.00001	0.00577		0.05647
12	+0.00011	0.00570		0.05491
15	+0.00017	0.01250		0.11459

## D3: Batch Size - 4

Total Params: 46781
Initial Loss: 4.0618
train_loss=tensor(2.5653), dev_loss=tensor(2.7414) and test_loss=tensor(2.7409)


### Tool 1

Layer  2/Tanh >> mean: 0.00296, std: 0.63813 with 0.00% saturation
Layer  5/Tanh >> mean: 0.00021, std: 0.62899 with 0.00% saturation
Layer  8/Tanh >> mean: -0.00049, std: 0.63091 with 0.00% saturation
Layer 11/Tanh >> mean: -0.00882, std: 0.63283 with 0.00% saturation
Layer 14/Tanh >> mean: -0.00256, std: 0.64411 with 0.00% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.01323
Layer  5/Tanh >> mean: -0.00000, std: 0.00734
Layer  8/Tanh >> mean: 0.00000, std: 0.00680
Layer 11/Tanh >> mean: 0.00000, std: 0.00552
Layer 14/Tanh >> mean: 0.00000, std: 0.00691

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00128	0.02270		0.09982
 3	-0.00007	0.00973		0.07042
 6	-0.00003	0.00739		0.06517
 9	-0.00001	0.00839		0.07784
12	-0.00000	0.00656		0.06254
15	+0.00012	0.01511		0.13380

## D3: Batch Size - 32

Total Params: 46781
Initial Loss: 3.6346

train_loss=tensor(2.2445), dev_loss=tensor(2.5393) and test_loss=tensor(2.5421)


### Tool 1

Layer  2/Tanh >> mean: -0.00558, std: 0.63759 with 2.41% saturation
Layer  5/Tanh >> mean: -0.00528, std: 0.64195 with 2.84% saturation
Layer  8/Tanh >> mean: -0.00283, std: 0.64306 with 2.22% saturation
Layer 11/Tanh >> mean: -0.00815, std: 0.64488 with 2.31% saturation
Layer 14/Tanh >> mean: -0.00627, std: 0.64767 with 2.09% saturation

### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00277
Layer  5/Tanh >> mean: -0.00000, std: 0.00238
Layer  8/Tanh >> mean: -0.00000, std: 0.00229
Layer 11/Tanh >> mean: 0.00000, std: 0.00217
Layer 14/Tanh >> mean: -0.00000, std: 0.00212


### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00000	0.01101		0.05893
 3	-0.00004	0.00742		0.07159
 6	-0.00000	0.00604		0.05858
 9	+0.00001	0.00577		0.05647
12	+0.00011	0.00570		0.05491
15	+0.00017	0.01250		0.11459

## D3: Batch Size - 256

Total Params: 46781
Initial Loss: 3.7875

train_loss=tensor(2.0750), dev_loss=tensor(2.4017) and test_loss=tensor(2.4237)

### Tool 1

Layer  2/Tanh >> mean: -0.00286, std: 0.63627 with 3.09% saturation
Layer  5/Tanh >> mean: 0.00007, std: 0.64129 with 3.45% saturation
Layer  8/Tanh >> mean: -0.00291, std: 0.64113 with 3.46% saturation
Layer 11/Tanh >> mean: -0.00399, std: 0.64390 with 3.16% saturation
Layer 14/Tanh >> mean: 0.00256, std: 0.64895 with 2.78% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00078
Layer  5/Tanh >> mean: 0.00000, std: 0.00073
Layer  8/Tanh >> mean: 0.00000, std: 0.00072
Layer 11/Tanh >> mean: -0.00000, std: 0.00065
Layer 14/Tanh >> mean: 0.00000, std: 0.00056

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00029	0.00963		0.05223
 3	-0.00007	0.00815		0.07997
 6	+0.00012	0.00781		0.07639
 9	-0.00005	0.00741		0.07301
12	+0.00005	0.00653		0.06333
15	+0.00002	0.01204		0.11485

# Part E

A4 and B3 as base

## 3 hidden layers

Total Params: 36581
Initial Loss: 3.8655
train_loss=tensor(2.2661), dev_loss=tensor(2.5638) and test_loss=tensor(2.5678)



### Tool 1

Layer  2/Tanh >> mean: -0.00887, std: 0.63561 with 2.53% saturation
Layer  5/Tanh >> mean: -0.00802, std: 0.64046 with 2.47% saturation
Layer  8/Tanh >> mean: -0.00873, std: 0.63781 with 2.38% saturation
Layer 11/Tanh >> mean: -0.00112, std: 0.64099 with 2.12% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00407
Layer  5/Tanh >> mean: 0.00000, std: 0.00355
Layer  8/Tanh >> mean: -0.00000, std: 0.00322
Layer 11/Tanh >> mean: -0.00000, std: 0.00292

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00015	0.00929		0.03040
 3	+0.00007	0.00778		0.04643
 6	+0.00001	0.00692		0.04114
 9	+0.00004	0.00616		0.03681
12	-0.00006	0.01045		0.06081


## 5 hidden layers

Total Params: 56981
Initial Loss: 4.0587
train_loss=tensor(2.2406), dev_loss=tensor(2.5405) and test_loss=tensor(2.5383)


### Tool 1

Layer  2/Tanh >> mean: -0.00377, std: 0.63115 with 2.59% saturation
Layer  5/Tanh >> mean: -0.00445, std: 0.63468 with 2.62% saturation
Layer  8/Tanh >> mean: -0.00409, std: 0.64165 with 2.47% saturation
Layer 11/Tanh >> mean: -0.00228, std: 0.64136 with 2.22% saturation
Layer 14/Tanh >> mean: 0.00303, std: 0.64666 with 2.03% saturation
Layer 17/Tanh >> mean: 0.00284, std: 0.64677 with 1.88% saturation


### Tool 2

Layer  2/Tanh >> mean: 0.00000, std: 0.00409
Layer  5/Tanh >> mean: -0.00000, std: 0.00349
Layer  8/Tanh >> mean: 0.00000, std: 0.00331
Layer 11/Tanh >> mean: 0.00000, std: 0.00294
Layer 14/Tanh >> mean: 0.00000, std: 0.00277
Layer 17/Tanh >> mean: -0.00000, std: 0.00272

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00006	0.01087		0.03554
 3	+0.00012	0.00795		0.04738
 6	+0.00000	0.00666		0.03954
 9	+0.00008	0.00625		0.03737
12	-0.00003	0.00550		0.03229
15	+0.00001	0.00541		0.03220
18	-0.00000	0.01035		0.05912

## 10 hidden layers

Total Params: 107981
Initial Loss: 3.5633
train_loss=tensor(2.2531), dev_loss=tensor(2.5363) and test_loss=tensor(2.5409)


### Tool 1

Layer  2/Tanh >> mean: 0.00051, std: 0.63204 with 2.59% saturation
Layer  5/Tanh >> mean: 0.00503, std: 0.63725 with 2.62% saturation
Layer  8/Tanh >> mean: -0.00086, std: 0.64806 with 2.19% saturation
Layer 11/Tanh >> mean: 0.00113, std: 0.64592 with 2.25% saturation
Layer 14/Tanh >> mean: -0.00187, std: 0.64586 with 1.88% saturation
Layer 17/Tanh >> mean: -0.00069, std: 0.64824 with 1.28% saturation
Layer 20/Tanh >> mean: 0.00147, std: 0.64555 with 2.06% saturation
Layer 23/Tanh >> mean: 0.00281, std: 0.64397 with 1.91% saturation
Layer 26/Tanh >> mean: -0.00152, std: 0.64738 with 1.66% saturation
Layer 29/Tanh >> mean: 0.00073, std: 0.64766 with 1.75% saturation
Layer 32/Tanh >> mean: 0.00404, std: 0.64642 with 1.50% saturation

### Tool 2

Layer  2/Tanh >> mean: -0.00000, std: 0.00331
Layer  5/Tanh >> mean: -0.00000, std: 0.00272
Layer  8/Tanh >> mean: -0.00000, std: 0.00267
Layer 11/Tanh >> mean: -0.00000, std: 0.00255
Layer 14/Tanh >> mean: -0.00000, std: 0.00263
Layer 17/Tanh >> mean: -0.00000, std: 0.00249
Layer 20/Tanh >> mean: -0.00000, std: 0.00246
Layer 23/Tanh >> mean: 0.00000, std: 0.00242
Layer 26/Tanh >> mean: 0.00000, std: 0.00237
Layer 29/Tanh >> mean: -0.00000, std: 0.00225
Layer 32/Tanh >> mean: -0.00000, std: 0.00235

### Tool 3

Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	+0.00005	0.00746		0.02438
 3	-0.00004	0.00566		0.03374
 6	+0.00004	0.00508		0.03013
 9	-0.00001	0.00507		0.03029
12	-0.00002	0.00492		0.02884
15	+0.00006	0.00518		0.03082
18	-0.00000	0.00503		0.02988
21	-0.00002	0.00475		0.02839
24	-0.00013	0.00460		0.02741
27	-0.00004	0.00459		0.02749
30	+0.00000	0.00450		0.02667
33	+0.00004	0.01018		0.06114

# Part takeaways

## Part A Takeaway

**A6 came out best, but only just.** A3, A4 and A6 all trained well, and the gaps between them are too small to call with one seed and 1000 steps. A1, A2 and A5 are clearly worse.

| Run | Init | Initial loss | Train / dev | Saturation | Tool 2 grad std, max/min | Hidden vs last-layer ratio (Tool 3) |
|---|---|---|---|---|---|---|
| A1 | std 1 | 19.26 | 3.98 / 4.57 | 90–93% | 11×, grows toward input | 0.013–0.13 vs 0.047 |
| A2 | ×0.1 | 6.71 | not logged | <1% | 6×, shrinks toward input | 0.12–0.20 vs 0.024 |
| A3 | 1/√fan_in | 3.74 | 2.244 / 2.563 | 1–4% | 1.19× | 0.060–0.071 vs 0.168 |
| A4 | ×5/3 | 4.17 | 2.206 / 2.565 | 14–21% | 1.17× | 0.039–0.055 vs 0.150 |
| A5 | ×3 | 5.66 | not logged | 51–55% | 1.57×, grows toward input | 0.031–0.043 vs 0.115 |
| **A6** | ×5/3, last ×0.1 | 3.85 | **2.200 / 2.550** | 13–20% | 1.17× | 0.037–0.053 vs 0.214 |

### Why A3, A4 and A6 are the good group

- In all three, the activation std is about the same in every layer: 0.56 for A3 and 0.74–0.76 for A4 and A6.
- The gradient std is flat across layers (within 1.2×), so the gradient flows cleanly in both directions.
- The hidden layers have similar grad/param ratios, and their Tool 4 lines sit together at about -2.2 to -2.4.
- Their dev losses are 2.563, 2.565 and 2.550. That 0.015 spread is noise-level, so A6 being lowest is not a proven win.

### Corrections to the per-run takeaways

- **A6, "updates faster, potential instability":** the last layer's high ratio is expected, not a sign of instability.
  - Its gradient is the same size as in A4 (about 0.024 in both). Its weights are just smaller, so the same gradient is a bigger relative step.
  - Tool 3 shows its weight std is 0.112 after the run. If it started near 0.017 (A4's 0.167 × 0.1), it grew about 7× in 1000 steps. So it is catching up, not unstable.
  - Its Tool 4 line is flat at about -1.65, with no spikes or growth. Instability would show up as spikes or a climbing line.
- **A6 #1:** shrinking the last layer's weights didn't lower its gradient magnitude. It raised its ratio, from 0.150 in A4 to 0.214.
- **A6 #2:** compare A6 with A4, its base, not with A5. Its saturation is the same as A4's (13–20%).
- **A5, "potential vanishing gradient":** Tool 2 shows the gradient growing toward the input (0.0055 to 0.0084), not vanishing. The real cost of A5 is that half the units are saturated, so they barely learn.
- **A5 heading:** it says "with BN", but the Tanh layers are at indices 1, 3, 5, 7 and 9, so there was no BN in this run. The heading looks like a typo.
- **A2:** the hidden-layer ratios are fairly even (0.12–0.20). The outlier is the last layer, which was still at std 1.

### The last-layer bias inflates the initial loss

A6 should start near ln(27) = 3.30, but it starts at 3.85, and A3 at 3.74. The last Linear layer has `bias=True` with `b = randn` (std 1) on line 70 of the py file. That gives the logits a random offset that the small weights can't hide, which plausibly accounts for most of the extra 0.5. Set the last layer's bias to zero (`layers[-1].b *= 0`) as part of the "low-confidence start", and A6 should start close to 3.30.

### Recommendation

- **Default:** use A6 (1/√fan_in, 5/3 gain, last layer scaled down) and zero the last-layer bias.
- **A3 versus A4:** A3 is just as viable here. The tanh gain made almost no difference to the dev loss at 5 layers, but A4's activation std was steadier from layer 1 onward. The gain should matter more with depth, so run E1 before deciding.
- **To separate A3, A4 and A6 for real:** run them for much longer (20k–50k steps, or the full schedule) or with 2–3 seeds. For init-only effects, also record Tools 1 to 3 at step 0.
- **Learning rate:** the hidden layers sit at about -2.3 in Tool 4, roughly 5× above the -3 target. `lr = 0.1` is a bit high, which Part C will show directly.

## Part B Takeaway

**With BN, the init stops changing the forward pass and changes the effective learning rate instead.**

| Run | Init | Initial loss | Train / dev | Tool 1 std / sat | Hidden ratio (Tool 3) | Hidden weight std | Step size, log10(lr × ratio) |
|---|---|---|---|---|---|---|---|
| B1 | std 1 | 3.63 | 2.577 / 2.810 | 0.63 / 2.5% | 0.0008–0.0013 | 0.99 | about -4.0 |
| B2 | ×0.1 | 3.63 | 2.288 / 2.594 | 0.64 / 2–3% | 0.055–0.071 | 0.103 | about -2.2 |
| B3 | 1/√fan_in | 3.63 | 2.287 / 2.598 | 0.64 / 2–3% | 0.055–0.072 | 0.104 | about -2.2 |
| B4 | ×5/3 | 3.63 | 2.293 / 2.630 | 0.64 / 2–3% | 0.031–0.042 | 0.168 | about -2.5 |
| B5 | B3, last bngain ×0.1 | 3.29 | 2.311 / 2.634 | 0.64 / 2–3% | 0.056–0.081 | 0.103 | about -2.2 |

(Hidden weight std = Tool 3 column ÷ ratio, for layer 3.)

### What changed what

- **BN fixes Tool 1, whatever the init.** All five runs show std 0.63–0.65 and 2–3% saturation, even B1, whose weights are 10× too big. BN rescales every pre-activation to mean 0 and std 1, so tanh always sees roughly N(0, 1). The tanh of a standard normal has std 0.628, and 3.6% of it lands beyond |0.97|. So "std ≈ 0.64 with 2–3% saturation" is BN's fingerprint. It only moves if `bngain` or `bnbias` learn away from 1 and 0 (see C4).
- **BN fixes the initial loss too.** B1 to B4 all start at exactly 3.6346, because the last BN normalizes the logits to std 1 no matter what the weights are. Logits with std 1 give about 3.63, not ln 27 = 3.30. With BN, the only knob for the starting confidence is the last `bngain`: B5 scales it by 0.1 and starts at 3.289.
- **Bigger weights → smaller relative updates, by the square.** If you multiply W by k, BN cancels it in the forward pass. So the gradient on W shrinks by k, while W itself grows by k. The grad/param ratio therefore falls by k².
  - B1 vs B3: hidden weights 10× bigger, so the prediction is 100× smaller. Observed: 0.0011 against 0.059, about 50×.
  - B4 vs B3: k = 5/3, so the prediction is 2.8× smaller. Observed: 1.7–2×.
  - Observed falls short of k² because the weights drift during 1000 steps.
- **So with BN, init scale is a hidden learning rate.** B1's forward pass looks perfect, but each step changes its weights by about 1/10,000 (Tool 4 sits at -3.6 to -4.2). That is why it has the worst loss of Part B (dev 2.81).
- **The tanh gain of 5/3 is useless with BN.** It exists to undo tanh's squashing when nothing renormalizes. BN already does that, so B4 has the same Tool 1 as B3 and only a lower effective lr (dev 2.630 against 2.598).
- **B2 is almost the same experiment as B3.** For the hidden layers (fan_in 100), 0.1 is exactly 1/√100. Only the first layer differs (0.1 against 1/√30 = 0.18), and that layer's ratio is 2.4× bigger in B2, as the k² rule predicts.
- **A low-confidence start (B5) only fixes the start.** It costs about 0.3 of loss at step 0 and is paid off within the first few dozen steps. Its dev loss is no better (2.634 against 2.598). It matters a lot more when the start is badly overconfident, as in A1 (19.3).

### Corrections to the per-run takeaways

- **B1, "W.grad std shrinks as we go deeper, vanishing gradients":** the ratio falls from layer 0 to layer 12, which is toward the **output**, so it is not vanishing gradients. Tool 2 grows only 1.6× toward the input. The real problem is the overall level: 0.001 × lr 0.1 = 1e-4 per step, 10× under the -3 target.
- **B2, "training at higher speed":** true compared with B1, but B2 trains at the same speed as B3 (see above).

## Part C Takeaway

**lr 0.1 was the best of the four after 1000 steps. Too low learns nothing, and too high (with BN) is worse but does not blow up.**

| Run | lr | Train / dev | Hidden ratio (Tool 3) | Step size, log10(lr × ratio) | Weight std: first / hidden / last (init 0.18 / 0.10 / 0.10) |
|---|---|---|---|---|---|
| C1 | 0.001 | 3.234 / 3.321 | 0.15–0.22 | about -3.7 | 0.18 / 0.099 / 0.100 |
| C2 | 0.01 | 2.697 / 2.867 | 0.12–0.19 | about -2.8 | 0.18 / 0.100 / 0.100 |
| C3 | 0.1 | 2.287 / 2.598 | 0.055–0.072 | about -2.2 | 0.19 / 0.104 / 0.109 |
| C4 | 1.0 | 2.374 / 2.771 | 0.014–0.017 | about -1.8 | 0.25 / 0.132 / 0.175 |

### What changed what

- **lr too low (C1): nothing moves.** The dev loss of 3.32 is barely better than a uniform guess (3.30). Tool 1 still looks exactly like init, and Tool 4 sits at -3.5 to -3.9.
- **lr ×10 does not mean step ×10 at the top end.** From 0.001 to 0.1 the step size rises about 1 log unit per 10× lr. From 0.1 to 1.0 it rises only 0.4 (2.5×). The reason is that the Tool 3 ratio itself falls as lr rises (0.2 → 0.06 → 0.015).
- **Why the ratio falls: with BN, big steps make the weights grow.** C4's weights grew 30–75% in 1000 steps, while C1 and C2 stayed at init. Bigger weights get smaller relative updates (the k² rule from Part B), so BN partly cancels a too-high lr by itself. This is why lr 1.0 did not diverge here.
- **lr too high (C4): noisier and worse, and BN's fingerprint breaks.** Dev 2.771 against 2.598. Tool 1 std rose to 0.67 and saturation in the first two tanh layers doubled (4.8% and 4.5% against about 2.4%). The big steps pushed `bngain` and `bnbias` away from 1 and 0. Tool 4 starts at about -0.5, which means the first steps change the weights by about 30% each.
- **The -3 rule is for long runs.** C2 is the closest to -3 (-2.8), but C3 (-2.2) has the best loss after 1000 steps. A short budget rewards a bigger step. The usual answer is a schedule: a high lr early and a decay later, which is what the `lr = 0.01` after step 100k in the code does.
- **Tool 3 alone misleads across learning rates.** C1 has the biggest ratio (0.27 on the last layer) but learns the least. Always multiply by lr, or look at Tool 4.

### Not done yet

The plan also says to repeat C1 and C4 on A3 (no BN). That is the run that would show how much of C4's survival is down to BN.

## Part D Takeaway

### D1: BN in only some layers

- **BN was removed from the first three hidden layers, not two.** Tanh layers 1, 3 and 5 follow a Linear layer directly, and the param count dropped by 600 = 3 × 200 BN params.
- **Without BN, those layers act like A3:** the std shrinks with depth (0.63 → 0.55 → 0.53). The first BN layer snaps it back to the 0.65 fingerprint.
- **Same loss as B3** (dev 2.597 against 2.598). With a good 1/√fan_in init and only 5 layers, BN adds nothing to the loss. A3, with no BN at all, got 2.563.
- **Inference:** BN is insurance against bad init (A1 dev 4.57 against B1 2.81) and against depth. When the init is already right, it mostly adds batch noise and a train/eval mismatch. Part E is where it should start to pay off.

### D2: BN momentum 0.1 vs 0.001

- **Tools 1 to 3 are identical to B3, digit for digit.** Momentum only updates `running_mean` and `running_var`, and training never reads them. Same batches, same gradients, same weights.
- **But all three losses changed** (train 2.287 → 2.245, dev 2.598 → 2.539). `get_loss` sets `training = False`, so even the "train loss" uses the running stats. The plan's expectation that the train loss stays the same doesn't hold in this code. What stays the same is the trained weights.
- **Why 0.001 won here:** with momentum 0.1, the running stats are an average of roughly the last 10 batches (about 320 examples), so they are noisy. With 0.001 they average about 1000 batches. The catch is that after 1000 steps, 0.999^1000 = 37% of the estimate is still the initial mean 0 and var 1. It won anyway, so the noise cost more than the stale part.
- **Inference:** momentum is purely an eval-time knob. A cleaner fix than tuning it is to compute the mean and var once over the whole training set after training.

### D3: batch size 4 vs 32 vs 256

| Batch | Examples seen (1000 steps) | Initial loss | Train / dev | Tool 1 saturation | Tool 2 grad std | Hidden ratio (Tool 3) |
|---|---|---|---|---|---|---|
| 4 | 4k | 4.06 | 2.565 / 2.741 | 0.00% | 0.0055–0.013 | 0.063–0.078 |
| 32 | 32k | 3.63 | 2.245 / 2.539 (momentum 0.001, see below) | 2–3% | 0.0021–0.0028 | 0.055–0.072 |
| 256 | 256k | 3.79 | 2.075 / 2.402 | 3–3.5% | 0.0006–0.0008 | 0.063–0.080 |

- **The batch-32 run is not B3.** Its numbers match D2 exactly, so it ran with momentum 0.001 left over from D2. The B3 numbers (dev 2.598) are the momentum-0.1 batch-32 result. Make sure the batch-4 and batch-256 runs used 0.1, or re-run them.
- **0% saturation at batch 4 is maths, not a healthy network.** With 4 samples, a batch-normalized value can never exceed (n − 1)/√n = 1.5 in size, and tanh(1.5) = 0.905, which is below the 0.97 threshold. With `bngain` near 1, saturation is impossible.
- **Tool 2 scales with 1/batch size, so don't compare it across batch sizes.** The loss is a mean over the batch, so each example's gradient is divided by B. Tool 3 sums over the batch, which is why its ratios match (0.06–0.08) at all three sizes.
- **A smaller batch gives a noisier gradient, not a smaller one.** The batch-4 Tool 4 has spikes of about ±1 log unit (10×) around a band from -2 to -2.5. The BN stats from 4 examples are just as noisy.
- **Batch 256 won by the most, but it saw 8× more data.** In 1000 steps it processed 256k examples against 32k. That's not a fair race. Compare at equal examples seen (for example, 125 steps at 256 against 1000 at 32), or give each run the same wall-clock time.
- **The initial loss differs because it comes from one batch.** With BN, the logits of that first batch depend on that batch's own stats. The batch-4 value is the noisiest.

## Part E Takeaway

All three runs use BN, a 5/3 gain, lr 0.1, momentum 0.1 and batch 32, which is the B4 setup. The weight std in Tool 3 (layer 3: 0.00778 / 0.04643 = 0.168) confirms the 5/3 gain. The labels count only the 100→100 Linear layers, so "3 hidden" has 4 tanh layers, "5" has 6, and "10" has 11. There is no run without BN (the A4 base), so the question "without BN, does the gradient vanish with depth?" is still open.

| Run | Tanh layers | Params | Train / dev | Tool 1 std / sat | Tool 2, first ÷ last tanh | Hidden ratio (Tool 3) |
|---|---|---|---|---|---|---|
| 3 hidden | 4 | 36,581 | 2.266 / 2.564 | 0.64 / 2.1–2.5% | 1.39× | 0.037–0.046 |
| 5 hidden | 6 | 56,981 | 2.241 / 2.541 | 0.63–0.65 / 1.9–2.6% | 1.50× | 0.032–0.047 |
| 10 hidden | 11 | 107,981 | 2.253 / 2.536 | 0.63–0.65 / 1.3–2.6% | 1.47× | 0.027–0.034 |

- **With BN, depth doesn't break the forward pass.** Every layer, even at 11 tanh layers, keeps the 0.64 std and a 1–3% saturation.
- **With BN, depth doesn't break the backward pass either.** The gradient grows only about 1.4–1.5× from the last tanh layer to the first, and that factor doesn't grow with depth. Nothing compounds layer by layer.
- **Deeper nets take slightly smaller steps.** The hidden ratio drifts down from about 0.04 to about 0.03, and inside the 11-layer net it falls slowly toward the output (0.034 → 0.027).
- **3× the params barely changed the loss.** Dev 2.564 → 2.541 → 2.536. At 1000 steps the model is limited by training time, not by size.
- **This is also a noise check.** B4 is the same setup with 5 tanh layers and got dev 2.630, worse than both its neighbours here (2.564 and 2.541). Changing the architecture changes the random stream, so a swing of 0.07–0.09 dev can come from the seed alone. Read every earlier difference of that size with care.

# Cause and effect summary

Everything below comes from one seed and 1000 steps. B4 against Part E shows that the seed alone can move the dev loss by up to about 0.09, so smaller differences (for example A3, A4 and A6, or D1 against B3) are too small to call.

## Weight init scale, no BN (Part A)

| Change | Effect | Evidence |
|---|---|---|
| Weight gain ↑ (1 → 5/3 → 3) | Saturation ↑ | 1–4% → 14–21% → 51–55% |
| | Activation std climbs toward 1 | 0.56 → 0.75 → 0.88 |
| | More overconfident start | Initial loss 3.74 → 4.17 → 5.66 |
| | Smaller relative updates | Hidden ratio 0.065 → 0.05 → 0.035 |
| | Gradients start growing toward the input | Tool 2 flat at 1 and 5/3, 1.6× at 3 |
| Weights far too big (std 1, about gain 10 at fan_in 100) | ~90% saturated in every layer | A1 |
| | Initial loss 19, still worse than uniform after 1000 steps | A1 |
| | Layers learn at very different speeds | Ratios spread 10×, gradients grow 11× toward the input |
| Weights too small (A2) | Activations shrink with depth, gradients shrink toward the input | 0.44 → 0.18, grad 0.028 → 0.0046 |
| Gain at 1/√fan_in × (1 to 5/3) | Flat activation std, flat Tool 2 | A3, A4, A6 |

A2 is not a clean "too small" test. 0.1 equals 1/√100 for the hidden layers, so only the first layer was small, and Tool 3 shows the last layer at std 0.97 (it looks unscaled).

## Last-layer scale

| Change | Effect | Evidence |
|---|---|---|
| Last `w` (no BN) or last `bngain` (BN) ×0.1 | Initial loss falls toward ln 27 = 3.30 | A6 3.85, B5 3.29 |
| | Its Tool 3 ratio rises (same gradient, smaller weights) | A4 0.150 → A6 0.214 |
| | Final loss: no reliable change | A6 −0.015 dev, B5 +0.036 dev |
| Last bias left at std 1 | Initial loss stays above 3.30 | A6 3.85 |

## BatchNorm

| Change | Effect | Evidence |
|---|---|---|
| Add BN | Tool 1 locked at std ≈ 0.64 and 2–3% saturation, whatever the init | B1–B5 |
| | Initial loss locked (3.63), whatever the init | B1–B4 |
| With BN, weights ×k | Relative update ÷k², so it acts like lr ÷k² | B1, B4 against B3 |
| With BN, tanh gain 5/3 | No forward effect, only a lower effective lr | B4 |
| With BN, lr too high | Weights grow and slow themselves down, so no blow-up | C4 |
| Remove BN with a good init (5 layers) | No change in loss | D1, A3 |
| BN momentum | No effect on training, only on eval (all three losses) | D2 |
| Smaller batch | Noisier BN stats and gradients; batch 4 caps saturation at 0 | D3 |

## Learning rate (with BN)

| Change | Effect | Evidence |
|---|---|---|
| lr ↓ (0.001) | Nothing learns; tools look like init | C1, dev 3.32 |
| lr ×10 | About +1 in Tool 4 at the low end, less at the high end | C1–C4 |
| lr ↑ (1.0) | Noisier, worse loss, BN params drift, saturation rises | C4 |
| Tool 4 at -3 | Not the fastest in a short run; -2.2 won at 1000 steps | C2 against C3 |

## Reading the tools

- A std near 1 means saturation, not health. Read it with the saturation %.
- A Tool 3 ratio is only meaningful × lr. Compare ratios across runs with the same lr only.
- Tool 2 values scale with 1/batch size.
- Tools 1 to 3 describe the network after 1000 steps, not at init.

## Bugs that affect every run

- **The embedding table `C` is never trained.** `params` is built from `layers` only, so `C` is left out (the "Total Params" count has no 270 for `C`). Every run so far used frozen random embeddings. This raises every loss by the same kind of amount, so comparisons between runs still hold, but the absolute numbers are worse than they should be. Fix: `params = [C] + [p for layer in layers for p in layer.parameters()]`.

# Inferences

![Deep NN diagnostics decision tree](../Notes/Month%204/Inferences%20decision%20tree.png)

Three rules behind the tree:

1. **Init scale decides saturation**, unless BN is there. Then BN fixes the activations, and the init scale only changes the step size (bigger weights → smaller steps).
2. **lr × Tool 3 ratio = the real step size.** Aim for around -3 in long runs, a bit higher in short ones.
3. **Differences under ~0.09 dev loss are seed noise** in these 1000-step runs.


## Session 2 — Initialization experiments

Code: [1 initialization.py](../Month%204/1%20initialization.py). Fixed setup: V = 27, block_size = 3, emb_size = 10, n_hidden = 200, batch = 32, 200k steps (lr 0.1, then 0.01 after 100k), seed 42 (`C`, `W1`, `b1`, `W2`, `b2` all drawn from the generator `g`).

Chain being initialized: `IDs → embeddings → concat → linear (W1, b1) → tanh → linear (W2, b2) → logits → loss`.

### Experiment A — Output confidence

**Expected initial loss.** With no information, the network should give every character the same probability 1/V. Then the loss is `-log(1/27) = 3.2958`. This is the target for step 0. A bigger value means the network is wrongly confident, not that it has learned something.

**Why the logits matter.** `logits = h1 @ W2 + b2`. With `W2 = randn`, the logits have a large spread. Softmax turns a large spread into a peaked distribution, and a confident wrong guess is penalised heavily.

| Output layer init | First mini-batch loss | Train / Dev / Test (after training) |
|---|---|---|
| `W2 = randn`, `b2 = randn` (unscaled) | **23.5189** | 1.9043 / 2.3265 / 2.3657 |
| `W2 = randn*0.01`, `b2 = 0` (small) | **3.3221** (≈ 3.2958) | 1.8449 / 2.3123 / 2.3599 |
| `W2 = 0` with a chosen bias (only `W2`, hidden layer random) | not run | not run |
| Everything zero (`W1, b1, W2, b2 = 0`) | 3.2958 | 2.6580 / 2.8156 / 2.8110 (see the zero-init break below) |

- The loss started at 23.5 instead of 3.3. Some early steps are spent only on shrinking the logits, which is wasted optimization.
- `b2 = 0` makes the output bias favour no class. `W2 * 0.01` makes `h1 @ W2` close to 0. Logits ≈ 0 means a uniform softmax, so the loss is ≈ log 27.
- It is not exactly 3.2958 because the logits are small but not exactly 0.
- `W2` is not set to exactly 0 in the code. Zero output weights are fine here (the gradient to `W2` is `h1ᵀ @ dlogits`, which is not 0 because `h1` is not 0), but the hidden layer must not also be zero (see the intentional break below).

### Experiment B — Tanh saturation

The first-layer scale is varied while the output layer uses the small init above.

- The pre-activation `embed @ W1` has std ≈ std(W1) × √30 × std(emb). With `W1 = randn` that is ≈ √30 ≈ 5.5.
- `tanh` of a large value is ±1, where the derivative `1 − tanh²(x)` is ≈ 0.
- A saturated unit passes ≈ 0 gradient back to `W1`, `b1` and `emb`. A neuron that is saturated for every example is dead and never learns.
- Saturation is measured as `|h1| > 0.97` after the first step (white pixels in the `imshow` plot).

| First-layer init | Saturated after step 0 | Train / Dev / Test |
|---|---|---|
| `W1 = randn` | 68.42 % | 1.8449 / 2.3123 / 2.3599 |
| `W1, b1 = randn*0.01` | 0 % (plot fully black) | 1.8866 / 2.2932 / 2.3144 |
| `W1 = randn * (5/3)/√30` | 18.5 % | 1.8089 / 2.2801 / 2.3234 |

- **`*0.01` removes saturation, but it is a magic constant.** The pre-activations are then nearly 0, tanh is almost linear there, and the layer barely uses its nonlinearity. The train loss is also worse than with `randn` (1.8866 vs 1.8449).
- **The 5/3 gain keeps a little saturation (18.5 %), which is acceptable.** Most units are in the curved region of tanh, so the nonlinearity is used, and the gradients still flow.

Not measured yet: the gradient std entering and leaving tanh, and the loss after an equal small number of updates (the table uses the loss after the full 200k steps).

### Experiment C — Fan-in scaling

Test: `x ~ N(0,1)` with shape (100, 200), multiplied by a (200, 100) weight matrix (fan-in 200).

| Weight scale | mean(xw) | std(xw) |
|---|---|---|
| `randn` (fixed std 1, ignores fan-in) | 0.087 | **14.31** |
| `randn * 0.2` (fixed std 0.2, ignores fan-in) | -0.009 | 2.81 |
| `randn / √200` (std ∝ 1/√fan-in) | -0.005 | **1.01** |

- `xw` is a sum of 200 terms, each of variance ≈ var(x)·var(w). Variances of independent terms add, so std(xw) ≈ √200 · std(w) · std(x) = 14.1 for std(w) = 1.
- `0.2` was a guess: 0.2 × 14.1 = 2.83, which matches the measurement. It did not reach 1, and it would be wrong for a layer with a different fan-in. The scale that does is `1/√fan_in`.
- The objective: the signal magnitude should not systematically grow or shrink from layer to layer. `1/√fan_in` is how that is achieved; it is not a constant to memorize.

**Gain for tanh.** tanh always squashes: even if its input has std 1, the output std is smaller. Stacked over layers the signal would keep shrinking toward 0. The fix is a gain, `std(W) = gain/√fan_in`. For tanh the gain is **5/3**, which is what `W1 * (5/3)/√(emb_size*block_size)` in the code does.

- Kaiming/He intuition: ReLU zeroes half the inputs, so its gain is √2. For tanh it is 5/3. The principle is the same: pick the gain that keeps the std stable.
- The code includes a check of `h = tanh(x @ w)` with gain 5/3. It prints mean and std for `x`, `xw` and `h`, and plots histograms for `x` and `xw`. Its output has not been logged in the notes yet.
- PyTorch's `torch.nn.init.kaiming_normal_` with `nonlinearity='tanh'` has not been compared to the hand-written gain version yet.

### Assignment M4.2 — Answers

1. **Which intervention corrected initial confidence?** Scaling the output layer: `W2 * 0.01` and `b2 = 0`. The first loss went from 23.52 to 3.32 (target 3.2958).
2. **Which corrected hidden saturation?** Shrinking `W1`: `*0.01` took saturation from 68.42 % to 0 %, and the fan-in/gain scaling `(5/3)/√30` took it to 18.5 %.
3. **Did either solve both problems automatically?** No. Scaling `W2` does nothing about `|h1|` being 68 % saturated, because it is applied after tanh. Scaling `W1` changes the hidden pre-activations but does not by itself fix the logits (`W2` must be small too). Both were needed, and they fix different places in the chain.
4. **Which measurement would have detected each problem fastest?** Overconfidence: the first-batch loss compared with `log(27) = 3.2958` (one forward pass). Saturation: the fraction of `|h1| > 0.97` after the first forward pass, or the std of the pre-activation.
5. **Why must controlled runs share data, seeds, architecture, and update count?** Otherwise a difference in loss cannot be attributed to the init. Here the dev loss only ranges from 2.2801 to 2.3265 across all four inits, so even small uncontrolled differences would be the same size as the effect. Even with everything shared, one seed is not enough to claim an ordering (see below).

### Evidence vs. explanation

- **Observed:** the 23.52 → 3.32 first loss, the 68.42 % → 0 % / 18.5 % saturation, and std 14.3 / 2.8 / 1.0.
- **Explained from these:** overconfident logits, saturation, and variance adding as a sum over fan-in.
- **Not shown by these runs:**
  - The final losses are single runs, one seed (42) each. Across the inits the dev loss only moves from 2.3265 to 2.2801, and test from 2.3657 to 2.3144. The ordering is not consistent: the 5/3 gain has the best dev (2.2801), but the `*0.01` init has the best test (2.3144). I cannot say that the ordering is real.
  - Train loss: 1.9043 for plain randn vs 1.8089 for the 5/3 gain, but `*0.01` has a worse train loss (1.8866) than the randn-`W1` run (1.8449) while its dev/test are better.
  - Train < dev everywhere. This is a model-capacity/overfit effect, not an init effect.
  - The effect of init is clearest in the first-step loss (23.5 → 3.3), where the gap is huge.
- Initialization matters more for deep networks. This MLP has only one hidden layer, so even a poor init still trains. In a deep stack the errors in scale compound with each layer.

### Intentional break — zero initialization

Setup: `W1`, `b1`, `W2`, `b2` all multiplied by 0, so every layer parameter is exactly 0. Only the embedding table `C` is still `randn`. Full 200k-step training, everything else unchanged.

| Init | Train / Dev / Test |
|---|---|
| All of `W1, b1, W2, b2 = 0` | **2.6580 / 2.8156 / 2.8110** |
| Best run above (5/3 gain) | 1.8089 / 2.2801 / 2.3234 |

The network trains (the loss is well below the uniform `log 27 = 3.2958`) but ends about 0.85 nats worse than the other inits. Why:

1. **Forward pass.** `l1 = embed @ 0 + 0 = 0`, so `h1 = tanh(0) = 0` for every unit and every example. The logits are `0 @ W2 + b2 = b2 = 0`, so the starting prediction is uniform and the first loss is `log 27`.
2. **Backward pass, output layer.** `dL/dW2 = h1ᵀ @ dlogits`. Since `h1 = 0`, this is exactly 0, so `W2` never changes and `h1` stays 0 for the entire run.
3. **Backward pass, hidden layer.** `dL/dh1 = dlogits @ W2ᵀ`. Since `W2 = 0`, this is exactly 0. So `dW1`, `db1` and `dC` are all exactly 0 as well (the gradient passes through `W2` before reaching them), so `W1`, `b1` and `C` never change.
4. **What does train:** only `b2`, because `dL/db2 = sum(dlogits)` does not depend on `h1`. The model reduces to a constant logit vector, which can learn only the overall frequency of each next character. Every input gives the same prediction.
5. So the loss 2.66 is expected to be about the entropy of the next-character distribution (a context-free, unigram model). Not yet verified against the character counts; I'd confirm it by computing that entropy directly from `ytrain`.

**Symmetry.** All hidden units are identical, not just similar: they all have the same incoming weights (0), the same bias, and the same outgoing weights (0), so they produce the same output (0) and receive the same gradient (0). They cannot specialize because nothing distinguishes them. Zero init is the extreme case: even if the symmetry were broken, there is no gradient to break it with. A non-zero constant init (e.g. every weight 0.1) would show the pure symmetry problem: the units would get gradients, but identical ones, and stay identical forever.

**Lesson.** Random init does two jobs here: it breaks the symmetry between hidden units, and it makes the signal and gradient non-zero so that learning can start. Scale controls the saturation and the confidence; randomness controls the symmetry. The observed gap (2.66 vs 1.81 train) comes entirely from the dead hidden layer, not from an optimizer problem.

---

## Takeaways (remember always)

- **Check the first loss.** It should be ≈ `log(V)` (`3.2958` for V = 27). A much bigger value (23.5 here) means the logits are overconfident, not that the optimizer is bad.
- **Make the output layer small:** `W2 * 0.01`, `b2 = 0`. Logits ≈ 0 gives a uniform softmax, so the loss starts at `log V` and no early steps are wasted shrinking the logits.
- **Saturated tanh passes no gradient.** The derivative is `1 − tanh²(x)`, which is ≈ 0 when `|x|` is large. Check the saturated fraction (`|h| > 0.97`), not just the loss.
- **Variances add.** A layer's output std is ≈ `√fan_in · std(W) · std(x)`. The fix is `std(W) = gain / √fan_in`, not a magic number like 0.01 or 0.2.
- **tanh gain = 5/3.** tanh squashes the signal, so the gain scales the weights up to compensate. Some saturation (about 18 %) is fine.
- **Mean ≈ 0 proves nothing.** Always look at the std too.
- **Don't init everything to zero.** `h1 = 0` kills the gradient to `W2`, and `W2 = 0` kills the gradient to `W1`, `b1` and `C`. Only `b2` trains, and the model becomes a unigram predictor (train 2.66 vs 1.81).
- **Scale and randomness do different jobs.** Scale controls saturation and confidence. Randomness breaks the symmetry between hidden units.
- **Init matters most in deep networks.** This one-hidden-layer MLP trains even with a bad init, so the final losses barely differ. The signal here is the first-step loss and the saturation.
- **One seed proves nothing.** The dev/test differences are about 0.03–0.05, so don't claim an ordering without more seeds. Make sure to use the generator everywhere in the code (which was missed in mini-batch sampling).

## Session 3,4

**How to think about init (from the [deep net runs](../Month%204/3%20deep%20nn%20runs.md)):** in a deep net, init is about keeping every layer's activation std and gradient std the same from layer to layer. `gain/√fan_in` does that for the hidden layers, and a small last layer gives a near-uniform first prediction. Diagnose it with the per-layer tools (saturation %, gradient std, grad/param ratio), not the final loss: in Part A the bad inits wrecked training (dev 4.57 for std-1) while every reasonable init ended within 0.015 of the others.

Other important files 
- [Inferences decision tree.png](../Notes/Month%204/Inferences%20decision%20tree.png)
- [3 deep nn runs.md](../Month%204/3%20deep%20nn%20runs.md)

## Session 5,6,7

Implemented the backprop by hand, but skipped the derivation of the optimisation of the loss's backward and bn layer's backward. Code in [4 backprop ninja.py](./../Month%204/4%20backprop%20ninja.py)

## Session 8,9

Answers are based on the active model in [5 wavenet.py:223-229](../Month%204/5%20wavenet.py#L223-L229).

**1. How many aggregation levels are needed to combine 8 positions pairwise?**
It takes 3 levels, because log₂(8) = 3. Each level halves the number of positions: 8 → 4 → 2 → 1. The model has three `FlattenConsecutive(2)` + `Linear` + `BatchNorm1D` + `Tanh` blocks, one per level.

**2. What is the receptive field after each level?**
- Level 1: 2 characters. Each output position sees 2 adjacent embeddings.
- Level 2: 4 characters. Each position combines two level-1 outputs.
- Level 3: 8 characters. This is the whole `block_size`, so it is the full context.

The receptive field doubles at each level.

**3. Why can the same transformation be reused across positions at one level?**
`FlattenConsecutive(2)` reshapes `(B, T, C)` into `(B, T/2, 2C)`. Each position then holds one pair, laid out the same way. `Linear` computes `x @ w` on the last axis only, so one weight matrix `(2C, n_hidden)` is applied to every pair. The positions share weights, as a conv with kernel size 2 and stride 2 would. This works for two reasons:
- Every pair has the same format, with the left and right element in the same slots.
- The rule for combining two neighbours does not depend on where in the word they sit.

`BatchNorm1D` also supports this. For 3D input it normalizes over `dim=(0,1)`, so batch and time share one set of statistics per channel.

**4. Why is hierarchical aggregation not yet attention?**
- **The grouping is fixed.** Only adjacent pairs `(0,1)`, `(2,3)`, … are combined. Position 0 can't look at position 5 until the last level, and only through the fixed tree.
- **The weights are content-independent.** `w` is the same whatever the characters are. Attention computes weights from the tokens themselves, through query·key similarity and a softmax.
- **There is no selective routing.** The model can't choose which earlier character matters for the current one. It can only mix them with learned, fixed linear maps.
- **The combination is concatenate then linear.** Attention takes a data-dependent weighted sum of value vectors.

So this is a fixed tree of shared-weight local mixers. It is like a dilated conv, not a content-addressed lookup.

**5. Which tensor axis represents time/space at every level?**
It is axis 1, the `T` in `(B, T, C)`. Axis 0 is the batch and the last axis is channels or features. Tracing the shapes:

| Step | Shape |
|---|---|
| Embedding | `(B, 8, 10)` |
| Flatten(2) → Linear | `(B, 4, 20)` → `(B, 4, 100)` |
| Flatten(2) → Linear | `(B, 2, 200)` → `(B, 2, 100)` |
| Flatten(2) → Linear | `(B, 1, 200)` → squeezed to `(B, 200)` → `(B, 100)` |
| Final Linear | `(B, 27)` |

The time axis shrinks 8 → 4 → 2 → 1 as it is folded into the channel axis. At level 3, `FlattenConsecutive` squeezes it away (lines 158-159), so the final `Linear` sees a 2D tensor.

## Session 10

Built the character-level pipeline on TinyShakespeare and trained a bigram baseline on it. Code is in [6 shakespeare.py](../Month%204/6%20shakespeare.py) and the [gpt package](../Month%204/gpt/): [tokenizer.py](../Month%204/gpt/tokenizer.py), [data.py](../Month%204/gpt/data.py), [bigram.py](../Month%204/gpt/bigram.py).

The goal is not a good model. It is a working data → loss → train → eval → generate loop, so that when attention comes in Session 11, any new bug can only be in the attention code.

**Pipeline, end to end**

```text
text → CharTokenizer.encode → 1-D long tensor → contiguous 90/10 split
     → get_batch → x [B,T], y [B,T]
     → BigramLanguageModel → logits [B,T,V]
     → reshape to [B*T,V] and [B*T] → cross_entropy → backward → Adam step
```

**Tokenizer** ([tokenizer.py](../Month%204/gpt/tokenizer.py))
- One token is one character. Capitalization, spaces and newlines are all kept, and the file is a single continuous stream with no word splitting or end-of-name token (this differs from the names dataset).
- The vocabulary is `sorted(set(text))`, with IDs `0..V-1`, stored in both `stoi` and `itos`. Sorting makes the mapping deterministic: the same text always gives the same IDs. A Python `set` has no stable order, so an unsorted vocab would give different IDs on different runs and break saved models.
- An unknown character raises `ValueError` by default. With `handle_unknown=True` it maps to `<UNK>`, which adds one entry to the vocab.
- `V` is built once from the full text and reused for both splits.

**Split and batching** ([data.py](../Month%204/gpt/data.py))
- `encode_and_split` cuts the encoded stream at `int(0.9 * N)`: the first 90% is train and the last 10% is validation. There is no shuffling. A random per-character split would put almost-identical overlapping windows in both sets and leak context into validation.
- `get_batch` picks `B` random start positions `i` and builds `x = data[i:i+T]`, `y = data[i+1:i+T+1]`. So `y` is `x` shifted by one, and `x[:, 1:] == y[:, :-1]`.
- `torch.randint(high=len(data) - T)` excludes `high`, so the largest start is `len - T - 1`. Then `y`'s slice ends exactly at `len`, and nothing runs off the end.
- A window needs `T+1` source characters, but `x` and `y` hold `T` each. Each row gives `T` prediction tasks, not one: `x[t] → y[t]` for every `t`.
- Sampling windows inside an already-separated region is fine. The leak is only when overlapping windows are assigned to train or validation at random.

**Model** ([bigram.py](../Month%204/gpt/bigram.py))
- `nn.Embedding(V, V)` is the entire model. Row `c` of the table is the next-character logits for input `c`. It is not a feature vector.
- Shapes: `[B,T]` long IDs → `[B,T,V]` logits. `nn.Embedding` needs integer IDs, so `torch.randn` input fails. Logits go out, never back in.
- Even with `T = 8`, each prediction uses one character. All `T` positions are computed in parallel, but they never see each other. The same input character gives the same logits in any sentence. `[B,T]` does not give sequence context by itself.

**Loss**
- `loss_fn(logits.view(-1, V), y.view(-1))`: `[B,T,V] → [B*T,V]` and `[B,T] → [B*T]`.
- With `B=32, T=8` that is 256 independent prediction tasks per step. `CrossEntropyLoss` wants the class (vocab) axis last in 2D (or at dim 1), so flatten `B` and `T` together and keep `V` as the last axis. The vocab axis must not be mixed into the batch axis, or the scores will be paired with the wrong targets.
- A fresh model should start at about `ln(V)`, the uniform guess. For 65 characters that is ≈ 4.17.

**Training and evaluation** (settings in [6 shakespeare.py](../Month%204/6%20shakespeare.py))
- seed 1337, `B=32`, `T=8`, Adam with `lr=1e-2`, 2000 steps, evaluating every 200.
- `estimate_loss` averages the loss over `eval_batches=20` fresh batches of `B=32`, with no gradients, then restores `train()`. A single batch would be noisy, so this gives a steadier number. `learning_rate` was originally defined but not passed to the optimizer, so it ran at Adam's default 1e-3. It is now passed in.
- The whole-stream check at the end runs `model(train[:-1])` against `train[1:]`. With 1-D input the embedding returns `[N,V]` directly, so the loss over all N tokens needs no batching.
- Final losses: **train 2.464, test 2.492**. That is down from ≈ 4.17, and validation is close to train, which is expected for a model with only `V×V` parameters. It is too small to overfit the training text.

**Generation**
- Loop: take the logits of the last position → softmax over the vocab axis → `torch.multinomial` → append the sampled ID → repeat. The output is `[B, T + max_new_tokens]`.
- It is sequential even though training is parallel: the next input does not exist until the previous one is sampled. A cache would not remove that dependency; it would only avoid recomputing old positions.
- The bigram model only uses the last token, so `generate` feeds just `idx[:, -1]`. That is shape `[B]`, the embedding returns `[B,V]`, and so softmax uses `dim=1`. Feeding the full prefix would work too but recomputes `T` positions it never uses.
- The method must call `self(...)`, not the global `model`. It only worked at first because the script's variable happened to be named `model`.
- Samples of 300 characters from a zero start (ID 0 is the first sorted character):

  | | Sample |
  |---|---|
  | Before training | `\nSKIcLT;AcELMoTbvZv C?nq-QE33:CJqkOKH-q;:la!oiywkHj...` (random characters, including rare ones like `$`, `3`, `&`) |
  | After training | `Thand, sowht ClanBy-r har / KINRD 'swind / We, Palion, s w pro y ben dis ounerrofrssple tim; ...` |

  After training you see speaker-label-like capitals, line breaks, common letter pairs (`th`, `ou`, `er`) and short word-like chunks. The words are not real, because each character depends only on the one before it.

**Takeaways**
- Check each piece alone before combining: tokenizer round-trip, shift (`y` is `x` moved by one), shapes, first loss ≈ `ln V`, then train.
- Random windows from a contiguous train region are good. Random assignment of overlapping windows to train and val is leakage.
- Bigram loss bottoms out around 2.4–2.5 for this data because the model sees one character. Lower needs context, so the next session adds communication between positions (attention). Everything built here is reused as is.


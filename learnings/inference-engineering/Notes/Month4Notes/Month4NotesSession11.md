[← Month 4 notes](<../Month4Notes.md>) · [← Session 10](<Month4NotesSession10.md>)

## Session 11 — Part 1: Raschka single-head walkthrough

Code: [7 sr singleheadattention.py](../../Month%204/7%20sr%20singleheadattention.py). Sentence: "Life is short, eat dessert first".

**Meaning of each step**

| Step | Code | Shape | Meaning |
|---|---|---|---|
| Embed | `embed_layer(encoded)` | `[6, 16]` | One vector per token. Carries no information about the other tokens yet. |
| Project | `query = embed @ W_query.T` (likewise key, value) | `[6, 24]`, `[6, 24]`, `[6, 28]` | The same embedding is viewed in three roles. Query: what this token is looking for. Key: what this token advertises so it can be found. Value: the payload it hands over when selected. |
| Score | `omega = query @ key.T` | `[6, 6]` | `omega[i, j] = query[i] · key[j]`, a *learned* similarity (similarity after projection, not raw embedding similarity). It is a score, not the output. |
| Scale + softmax | `softmax(omega / sqrt(dk), dim=1)` | `[6, 6]` | Turns each row into a distribution over key positions: how the token `i` spends its attention budget. Each row sums to 1. Softmax runs over the key axis because the question is "which keys?" |
| Mix values | `attention_weights @ value` | `[6, 28]` | `context[i] = Σ_j w[i,j] · value[j]`. A weighted average of value vectors, giving a new, context-enriched vector for every token. |

One-line summary: query-key decides *where* to look, value decides *what* to take, and the context vector is the token's embedding rewritten using what it looked at.

**Convention rewrite** (article is column-vector, this curriculum is row-vector)

```text
Article:     q = W_q @ x        W_q: [d_q, d]      (code: embed @ W_query.T)
Curriculum:  q = x @ Wq         Wq:  [C, D]        (Wq = W_query.T)
```

**Differences from this curriculum**

- No causal mask. The `[6, 6]` matrix is full, so token 1 attends to tokens 2 to 5 as well. This is the encoder (BERT-style) special case. Causal attention masks `j > i` before the softmax.
- Unbatched `[T, d]` rather than `[B, T, C]`.
- Weights are `torch.rand` (all positive), not zero-mean.

**Experiments and findings**

1. **Shapes.** `W_query/W_key: [24, 16]`, `W_value: [28, 16]`; `query/key: [6, 24]`, `value: [6, 28]`; `omega` and weights `[6, 6]`; context `[6, 28]`. `dv != dk` works because values never enter the dot product; only `dq == dk` is required.
2. **First run did not match Raschka, even though the embeddings were identical** (`nn.Embedding(6, 16)`, seed 123). The cause was the weight initialization: I used `torch.randn` (normal, mean 0, std 1) and he uses `torch.rand` (uniform on [0, 1)). The same seed with a different sampler gives different numbers. After switching to `torch.rand` the outputs match his exactly.
3. **Reproduced Raschka's numbers.** `omega[1] = [8.5808, -7.6597, 3.2558, 1.0395, 11.1466, -0.4800]`, and `omega_24 = query[1] · key[4] = 11.1466`. Weights `[0.2912, 0.0106, 0.0982, 0.0625, 0.4917, 0.0458]`. Context vector shape `[6, 28]`.
4. **Reading the result.** Sorted vocabulary gives `Life=0, dessert=1, eat=2, first=3, is=4, short=5`, so the encoded sentence is `[0, 4, 5, 2, 1, 3]`. Row 1 is "is". It puts the most weight on "dessert" (0.49) and then "Life" (0.29). These weights come from random matrices, so they have no linguistic meaning yet. Training is what makes them meaningful.
5. **`randn` vs `rand`, the scale lesson.** With `randn` the raw scores were about ±100 and the softmax row was nearly one-hot: `[0.074, 0.924, 3.6e-18, 1.4e-3, 6.3e-18, 2.2e-8]`. With `rand` the maximum weight is 0.49. A saturated softmax has near-zero gradients on the non-winning entries, so it starves `Wq` and `Wk` of learning signal (the Sessions 1-2 saturation story).
6. **Why dividing by `sqrt(dk)` was not enough in the `randn` run.** The `sqrt(D)` argument assumes `q` and `k` have unit-variance components. Here both `x` and the weights are N(0, 1), so each `q` component has variance about C = 16, not 1. The score std becomes about `C * sqrt(D) = 16 * 4.9`, then `/ sqrt(D)` leaves a std of about 16. The scaling fixes only the `D` factor. A real model must also initialize the weights with std about `1/sqrt(C)` so that `q` and `k` start near unit variance. This feeds into Part 3.

**Open items for Session 11**

- Part 3: derive `Var(q·k) = D` and measure the four numbers.
- Part 4: shape ledger including the three loops.
- Parts 5-8: hand example, `loop_attention`, the six invariants, break-it log.

## Session 11 — Part 2: Concepts before code

The search-engine analogy: **query** is what a position is looking for, **key** is what each position advertises so it can be found, **value** is the payload delivered once a position is selected. It is intuition. It does not guarantee what a trained head learns.

**1. What question does a query represent computationally?**
A query is a vector in the same space as the keys. The question it asks is "which positions have keys that point in my direction?". It is answered by taking a dot product with every key. The result is one score per position, so the query produces a row of `T` scores (`q[i] · k[j]` for each `j`) and not an answer itself.

**2. What does a key do differently from a value?**
- A key is used only for **matching**. It decides *how much* attention a position receives, because it only enters the scores.
- A value is used only for **content**. It decides *what* flows into the output once the weights are fixed, because it only enters the weighted sum.
- They need separate vectors because "easy to find" and "useful to receive" are different jobs. The same position can have a key that says "I am a noun" and a value that carries what that noun means. This is also why `Dv` can differ from `D`: values never enter the dot product, so only `Dq == Dk` is required (see the run in Part 1: `dv = 28`, `dk = 24`).

**3. Why are query-key dot products scores rather than outputs?**
- A dot product is a single scalar per pair, shape `[T, T]`. It measures relevance but carries no content.
- It is unbounded and can be negative, so it cannot be used as a mixing weight directly. Softmax turns it into non-negative weights that sum to 1.
- The output has to live in value space, `[T, Dv]`. It is created by using the normalized scores to average the values.
- So the pipeline is score → normalize → mix. The scores are only the decision about where to look.

**Why not use the raw `x` for all three roles?**
If `q = k = v = x`, one vector has to be a good search term, a good index entry and a good payload at once. Two concrete problems:
- **The score matrix is symmetric.** `x_i · x_j = x_j · x_i`, so "i attends to j" always equals "j attends to i". Relations such as a verb looking for its subject are directional, and that cannot be expressed.
- **Self-match dominates.** For vectors of similar norm, `x_i · x_i = |x_i|²` is the largest score in the row, so every token mostly attends to itself and the output barely changes from the input.

With separate projections, `q[i] · k[j] = x_i Wq Wk^T x_j^T`, and `Wq Wk^T` is generally not symmetric, so directional relations become possible. Each projection can also be trained for its own job.

**Evidence from my run (random weights, no training):** in `attention_weights[1]`, token 1 ("is") gives itself only 0.0106 and gives "dessert" 0.4917. Without the projections, the self-score would be the biggest. This shows that the projections already break the self-match, though nothing has been learned yet.

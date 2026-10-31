### Objective


Build a clean character-level sequence dataset and reproduce the simple baseline that the GPT lecture evolves into attention.


### Data


Use TinyShakespeare from the source linked by the GPT lecture repository:


<https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt>


The lecture's reference GPT uses the TinyShakespeare text, a character vocabulary, random fixed-length chunks, and shifted next-character targets.[cite:189]


### Character tokenizer


Implement:


```python
class CharTokenizer:
    encode(text) -> list[int]
    decode(ids) -> str
    vocab_size
```


Tests:


- `decode(encode(text)) == text` for representative text.
- All encoded IDs are in range.
- Vocabulary order is deterministic.
- Unknown-character behavior is explicit.


### Contiguous split


Split the encoded stream into contiguous training and validation regions. Do not randomly distribute individual characters between splits, because overlapping sequence windows could otherwise leak nearly identical local context across them.


### Batch sampler


Given `block_size = T` and `batch_size = B`, return:


```text
x: [B, T]
y: [B, T]
```


For every row:


```text
x = data[i : i + T]
y = data[i + 1 : i + T + 1]
```


Write a manual decoded example proving the one-token shift.


### Baseline model


Implement a token-transition model that maps each input token directly to logits over the next token at every time position. Train it briefly and generate text. Its purpose is to validate the data pipeline, loss flattening, optimizer, evaluation, and generation loop before attention is introduced.


### Assignment M4.10


Answer:
 

1. Why does each sequence provide `T` next-token training examples?
2. Why can all `T` losses be computed in parallel during training?
3. Why must generation still proceed one token at a time without a cache?
4. Why is random-window sampling different from randomly splitting overlapping windows?
5. Why must logits `[B, T, V]` and targets `[B, T]` be reshaped for cross-entropy?


### Deliverable


- `gpt/tokenizer.py`.
- `gpt/data.py`.
- Tokenizer and batch tests.
- Baseline training and sample.


### Exit criteria


- Encode/decode round-trip passes.
- Every target is the next token after the corresponding input.
- The baseline overfits a tiny batch and learns on the training stream.
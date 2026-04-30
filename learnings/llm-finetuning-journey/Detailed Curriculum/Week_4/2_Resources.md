# Week 4 — Resources

## Videos

- [Recurrent Neural Networks (RNN) and LSTM Clearly Explained](https://www.youtube.com/watch?v=AsNTP8Kwu80) — StatQuest with Josh Starmer, YouTube, 16m. Clean, accessible walkthrough of the RNN computation.
- [Long Short-Term Memory (LSTM) Clearly Explained](https://www.youtube.com/watch?v=YCzL96nL7j0) — StatQuest, YouTube, 20m. Visual walkthrough of all four gates.
- [Illustrated Guide to LSTMs and GRUs: A step by step explanation](https://www.youtube.com/watch?v=8HyCNIVRbSU) — Michael Phi, YouTube, 13m. Good visual summary before implementation.

## Blog Posts / Articles

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — Chris Olah. **Required reading.** The clearest explanation of LSTM gates in existence. Read this before the StatQuest videos.
- [The Unreasonable Effectiveness of Recurrent Neural Networks](https://karpathy.github.io/2015/05/21/rnn-effectiveness/) — Andrej Karpathy. Historical context: what RNNs could do before transformers. Read Sections 1–3.
- [An Introduction to Recurrent Neural Networks and the Math That Powers Them](https://machinelearningmastery.com/an-introduction-to-recurrent-neural-networks-and-the-math-that-powers-them/) — Jason Brownlee. Mathematical supplement to Olah's post.

## Papers

- [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — Hochreiter & Schmidhuber, Neural Computation 1997. The original LSTM paper. Sections 1–3 for historical context; the equations in Section 2 are what you implemented.
- [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — Cho et al., 2014. Introduces the GRU. Read the GRU description in Section 2.

## Documentation

- [nn.LSTM docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html) — Study the input/output shapes carefully. Note the `batch_first` parameter and the `h_n`, `c_n` output shapes.
- [nn.LSTMCell docs](https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html) — Single step LSTM; useful for implementing custom recurrent logic.
- [torch.nn.utils.clip_grad_norm_](https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html) — Gradient clipping. Always use for RNN training.

## GitHub Repos

- [karpathy/char-rnn](https://github.com/karpathy/char-rnn) — The original character-level RNN (in Lua/Torch, historical). The README is still a great read for understanding what char-RNNs learn.
- [spro/practical-pytorch](https://github.com/spro/practical-pytorch) — Practical PyTorch tutorials including char-RNN and LSTM sequence generation. Good reference implementations.

## Optional / Bonus

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — Sutskever et al., 2014. The seq2seq paper — introduced encoder-decoder LSTMs for machine translation. Read the abstract + Section 2. This is the architecture that attention was designed to fix.
- [On the difficulty of training Recurrent Neural Networks](https://arxiv.org/abs/1211.5063) — Pascanu et al., 2013. The paper that formally analyzes vanishing and exploding gradients in RNNs and introduces gradient clipping. Read Sections 1–2.
- [An Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling](https://arxiv.org/abs/1412.3555) — Chung et al., 2014. Head-to-head comparison of LSTM vs. GRU. The result: performance is similar; choose GRU for speed.

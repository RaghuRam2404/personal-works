# Week 9 Resources

## Papers

- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — Bahdanau et al. 2014. The original attention paper. Read fully and take handwritten notes.
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — Luong et al. 2015. Introduces multiplicative (dot-product) attention as a simpler alternative to Bahdanau.
- [On the Properties of Neural Machine Translation: Encoder-Decoder Approaches](https://arxiv.org/abs/1409.1259) — Cho et al. 2014. The vanilla seq2seq paper that Bahdanau was responding to.

## Videos

- [Yannic Kilcher — Attention Is All You Need (first 10 min only this week)](https://www.youtube.com/watch?v=iDulhoQ2pro) — Yannic Kilcher, 28 min total. Watch only the historical intro this week.
- [Sequence to Sequence Learning with Neural Networks — Illustrated](https://www.youtube.com/watch?v=L8HKweZIOmg) — Serrano.Academy, ~20 min. Good visual walkthrough of vanilla seq2seq before attention.

## Blog Posts / Articles

- [Attention? Attention!](https://lilianweng.github.io/posts/2018-06-24-attention/) — Lilian Weng. The canonical blog post on the history of attention mechanisms. Read sections 1–4.
- [Visualizing A Neural Machine Translation Model](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/) — Jay Alammar. Excellent animated visualization of how attention weights move during translation.

## GitHub Repos

- [bentrevett/pytorch-seq2seq](https://github.com/bentrevett/pytorch-seq2seq) — Well-structured PyTorch implementations of multiple seq2seq variants including Bahdanau attention. Use as reference only after attempting yourself.
- [karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) — Preview for Week 11. Note the contrast between RNN-based attention here and the pure-transformer approach.

## Documentation

- [PyTorch nn.GRU documentation](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html) — Pay attention to the `batch_first` parameter and the shape of the hidden state output.
- [torch.nn.utils.rnn.pack_padded_sequence](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pack_padded_sequence.html) — For efficient handling of variable-length sequences. Optional but good to know.

## Optional / Bonus

- [Effective Attention Visualization with matplotlib](https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html) — Official matplotlib tutorial for annotated heatmaps.
- [The Illustrated BERT, ELMo, and co.](https://jalammar.github.io/illustrated-bert/) — Jay Alammar. Connects this week's attention to the BERT era (preview of Phase 3+).

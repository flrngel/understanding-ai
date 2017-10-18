# Understanding process of Attention Mechanism in NMT

Understanding how attention (Bahdanau, Luong) works in NMT.

Tensorflow has fixed BeamSearchDecoder recently but not have been released yet. [check commit](https://github.com/tensorflow/tensorflow/commit/18f89c81d288f191abd56501ec6f86fe29265bdd#diff-4c273149acffdccde2f59be808317b3e)

So this code has to be test after 1.4 release.

Past commit includes working with BasicDecoder (Greedy)

## Todo

- [ ] test BeamSearchDecoder with tensorflow 1.4 
- [ ] make BLEU metrics work

## Paper References

- [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215)
- [Massive Exploration of Neural Machine Translation Architectures](https://arxiv.org/abs/1703.03906)
- [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025)
- [Sequence-to-Sequence Learning as Beam-Search Optimization](https://arxiv.org/pdf/1606.02960.pdf)

## Other References

- [https://harvardnlp.github.io/seq2seq-talk/slidescmu.pdf]
- [https://www.youtube.com/watch?v=UXW6Cs82UKo]

## Code References

- [https://github.com/tensorflow/tensorflow] (Attention test codes, Beamsearch test codes)
- [https://github.com/tensorflow/nmt]
- [https://github.com/JayParks/tf-seq2seq]
- [https://github.com/piyank22/Blog/blob/master/TheJourneyOfNLP.ipynb]
- [https://github.com/pplantinga/tensorflow-examples]

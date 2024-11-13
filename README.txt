Math to LaTeX.
End goal, easy screenshot complicated typed math equation, software then converts to LaTeX code to get that equation.
This problem has been solved. Most services charge for it.
Multi-purpose chatbots like ChatGPT and Claude can also do this.

Research papers on the topic:
Title: Translating Math Formula Images to LaTeX Sequences Using Deep Neural Networks with Sequence-level Training
Authors: Zelun Wang, Jyh-Charn Liu
Year: 2019
Link: https://arxiv.org/pdf/1908.11415
Overview: Neural Network architecture is CNN-LSTM (encoder-decoder)
Encoder (CNN) transforms images into group of feature maps.
Feature maps augmented with 2D positional encoding then unfolded into vector.
Decoder (LSTM stacked bidirectional) soft attention mech, translates encoder output into seq of LaTeX tokens.
Training is two steps.
First token-level training using MLE as obj func.
Then, seq-level training obj func used to optimize overall model based on policy gradient algo (reinforcement learning)
Overcomes bias exposure problem by closing feedback loop in decoder during seq-level training.
Trained and evaluated on IM2LATEX-100K dataset.

IM2LATEX-100K: https://www.kaggle.com/datasets/shahrukhkhan/im2latex100k
more info: https://zenodo.org/records/56198#.YHM2xRQzbvd

other reseach papers:
1. the original (i think) Harvard: https://arxiv.org/pdf/1609.04938v1
github: https://github.com/harvardnlp/im2markup
2. this one builds on above Stanford: https://cs231n.stanford.edu/reports/2017/pdfs/815.pdf
github: https://github.com/guillaumegenthial/im2latex

on topic of tokenizing / token vocab:

https://github.com/mathematicator-core/tokenizer
https://github.com/Miffyli/im2latex-dataset


complete project: https://github.com/lukas-blecher/LaTeX-OCR

# https://github.com/lukas-blecher/LaTeX-OCR/blob/main/pix2tex/models/transformer.py


# Question Answering Using Semantic Roles (PropQA-Net) NLP Project

This repository implements **PropQA-Net**, a classical (non-transformer) question answering system *anchored in semantic roles*. It is trained on real **PropBank** annotations (via **NLTK**) and learns two tasks jointly:

- **SRL tagging** (BIO tags over a sentence: `B-ARG0`, `I-ARG1`, `O`, ...)
- **Extractive QA** (predict an answer span inside the sentence)

# ☂️ BoCoEL

### Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models

![Logo](assets/logo-full.svg)

[![Publish](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml)
[![Build Pages](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml)
[![Formatting](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml)
[![Type Checking](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml)
[![Unit Testing](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml)

![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bocoel)
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

### 🤔 Why BoCoEL?

Evaluating large language models are expensive and slow, and the size of modern datasets are gigantic. If only there is a way to just select a meaningful subset of the corpus and obtain a highly accurate evaluation.....

Wait, sounds like [Bayesian Optmization](#bayesian-optimization)!

This library encodes

### 🚀 Features

- 🎯 Accurately evaluate large language models with just tens of samples from your selected corpus.
- 💂‍♂️ Uses the power of Bayesian optimization to select an optimal set of samples for language model to evaluate.
- 💯 Evalutes the corpus on the model in addition to evaluating the model on corpus.
- 🤗 Integration with huggingface [transformers](https://huggingface.co/docs/transformers/en/index) and [datasets](https://huggingface.co/docs/datasets/en/index)
- 🧩 Modular design.

### 🚧 TODO: work in progress

- 📊 Visualization module of the evaluation.
- 🎲 Integration of alternative methods (random, kmedoids...) with Gaussian process.

### Bayesian Optimization

> Image from wikipedia
![](https://upload.wikimedia.org/wikipedia/commons/0/02/GpParBayesAnimationSmall.gif)

Simply put, Bayesian optimization aims to optimize either the exploration objective (the purple area in the image) or the exploitation object (the height of the black dots). It uses Gaussian processes as a backbone for inference, and uses an **acquisition function** to decide where to sample next. See [here](https://distill.pub/2019/visual-exploration-gaussian-processes/) for an a more in-depth introduction.

### Bayesian Optimization in this Project

TODO: 

### ⬇️ Installation

I don't want optional dependencies:

```
pip install bocoel
```

Give me the full experience (all optional dependencies):

```
pip install "bocoel[all]"
```

## ⭐ Give me a star!

If you like what you see, please consider giving this a star (★)!

## 🥰 Contributing and Using

Openness and inclusiveness are taken very seriously. The code is available under [Apache License](./LICENSE.md). Please follow the guide to [contributing](./CONTRIBUTING.md) and the [code of conduct](./CODE_OF_CONDUCT.md).

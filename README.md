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

### ⬇️ Installation

I don't want optional dependencies:

```
pip install bocoel
```

Give me the full experience (all optional dependencies):

```
pip install "bocoel[all]"
```

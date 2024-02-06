# ☂️ BoCoEL

## Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models

![Logo](assets/logo-full.svg)

[![Publish](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml)
[![Build Pages](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml)
[![Formatting](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml)
[![Type Checking](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml)
[![Unit Testing](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml)


![GitHub License](https://img.shields.io/github/license/:user/:repo)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/bocoel)
[![Built with Material for MkDocs](https://img.shields.io/badge/Material_for_MkDocs-526CFE?style=for-the-badge&logo=MaterialForMkDocs&logoColor=white)](https://squidfunk.github.io/mkdocs-material/)

## 🤔 Why BoCoEL?

Large language models are expensive and slow behemoths, and evaluating them on gigantic modern datasets only makes it worse. 

If only there is a way to just select a meaningful (_and small_) subset of the corpus and obtain a highly accurate evaluation.....

Wait, sounds like [Bayesian Optmization](#bo)!

Bocoel works in the following steps:

1. Encode individual entry into embeddings (way cheaper / faster than LLM and reusable).
2. Use Bayesian optimization to select queries to evaluate.
3. Use the queries to retrieve from our corpus (with the encoded embeddings).
4. Profit.

The evaluations generated are easily managed by the provided manager utility.

## 🚀 Features

- 🎯 Accurately evaluate large language models with just tens of samples from your selected corpus.
- 💂‍♂️ Uses the power of Bayesian optimization to select an optimal set of samples for language model to evaluate.
- 💯 Evalutes the corpus on the model in addition to evaluating the model on corpus.
- 🤗 Integration with huggingface [transformers](https://huggingface.co/docs/transformers/en/index) and [datasets](https://huggingface.co/docs/datasets/en/index)
- 🧩 Modular design.

## 🗺️ Roadmap: work in progress

- 📊 Visualization module of the evaluation.
- 🎲 Integration of alternative methods (random, kmedoids...) with Gaussian process.
- 🥨 Integration with more backends such as [VLLM](https://github.com/vllm-project/vllm) and [OpenAI's API](https://github.com/openai/openai-python).

## ⭐ Give us a star!

Like what you see? Please consider giving this a star (★)!

## <a id="bo"></a> ♾️ Bayesian Optimization

<img src="https://upload.wikimedia.org/wikipedia/commons/0/02/GpParBayesAnimationSmall.gif" width="40%" align="right"/>

Simply put, Bayesian optimization aims to optimize either the exploration objective (the purple area in the image) or the exploitation object (the height of the black dots). It uses Gaussian processes as a backbone for inference, and uses an **acquisition function** to decide where to sample next. See [here](https://distill.pub/2019/visual-exploration-gaussian-processes/) for an a more in-depth introduction.

Since _Bayesian optimization works well with expensive-to-evaluate black-box model (paraphrase: LLM)_, it is perfect for this particular use case. Bocoel uses Bayesian optimization as a backbone for exploring the embedding space given by our corpus, which allows it to select a good subset acting as a mini snapshot of the corpus.


## ⬇️ Installation

I don't want optional dependencies:

```
pip install bocoel
```

Give me the full experience (all optional dependencies):

```
pip install "bocoel[all]"
```


## 🥰 Contributing

Openness and inclusiveness are taken very seriously. Please follow the guide to [contributing](./CONTRIBUTING.md) and the [code of conduct](./CODE_OF_CONDUCT.md).

## 🏷️ License and Citation

The code is available under [Apache License](./LICENSE.md).

TODO: Citation

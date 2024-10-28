# ☂️ BoCoEL

## Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models

![Logo](assets/logo-full.svg)

[![Publish](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/release.yaml)
[![Build Pages](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/build.yaml)
[![Formatting](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/format.yaml)
[![Type Checking](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/typecheck.yaml)
[![Unit Testing](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml/badge.svg)](https://github.com/rentruewang/bocoel/actions/workflows/unittest.yaml)


![GitHub License](https://img.shields.io/github/license/rentruewang/bocoel)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)


## 🤔 Why BoCoEL?

Large language models are expensive and slow behemoths, and evaluating them on gigantic modern datasets only makes it worse.

If only there is a way to just select a meaningful (_and small_) subset of the corpus and obtain a highly accurate evaluation.....

Wait, sounds like Bayesian Optimization!

Bocoel works in the following steps:

1. Encode individual entry into embeddings (way cheaper / faster than LLM and reusable).
2. Use Bayesian optimization to select queries to evaluate.
3. Use the queries to retrieve from our corpus (with the encoded embeddings).
4. Profit.

The evaluations generated are easily managed by the provided manager utility.

To our knowledge, this is the first work aiming to reduce computation costs during evaluation (benchmarking) with a (possibly dynamic) budget.


## 🚀 Features

- 🎯 Accurately evaluate large language models with just tens of samples from your selected corpus.
- 💂‍♂️ Uses the power of Bayesian optimization to select an optimal subset of samples for the language model to evaluate.
- 💯 Evaluate the corpus on the model in addition to evaluating the model on the corpus.
- 🤗 Support for `GPT2`, `Pythia`, `LLAMA` and more through integration with huggingface [transformers](https://huggingface.co/docs/transformers/en/index) and [datasets](https://huggingface.co/docs/datasets/en/index)
- 🧩 Modular design.
- 🔎 Efficient representation of the corpus / dataset such as N-sphere representation or whitening of the latent space to augment evaluation quality.


## ⭐ Give us a star!

Like what you see? Please consider giving this a star (★)!


## ♾️ Bayesian Optimization

<img src="https://upload.wikimedia.org/wikipedia/commons/0/02/GpParBayesAnimationSmall.gif" width="30%" align="right"/>

Simply put, Bayesian optimization aims to optimize either the exploration objective (the purple area in the image) or the exploitation object (the height of the black dots). It uses Gaussian processes as a backbone for inference, and uses an **acquisition function** to decide where to sample next. See [here](https://distill.pub/2019/visual-exploration-gaussian-processes/) for an a more in-depth introduction.

Since _Bayesian optimization works well with an expensive-to-evaluate black-box model (paraphrase: LLM)_, it is perfect for this particular use case. Bocoel uses Bayesian optimization as a backbone for exploring the embedding space given by our corpus, which allows it to select a good subset acting as a mini snapshot of the corpus.


## 🏎️ Performance Implications

LLMs are painfully slow, especially generative ones (which is what is usually referred to as LLM), since sequence generation is sequential by nature.

Despite `bocoel`'s requirement to use an embedder to encode the entire corpus, embedders are faster than LLMs by orders of magnitude and the time is gained back by practically any savings in evaluating LLMs.


## ⬇️ Installation

I don't want optional dependencies:

```
pip install bocoel
```

Give me the full experience (all optional dependencies):

```
pip install "bocoel[all]"
```


## 🔬 Usage

See the folder [examples/getting_started](https://github.com/rentruewang/bocoel/tree/main/examples/getting_started) for a simplistic usage of the library to get started with just a few lines of code.


## ✍️ Develop with BoCoEL

Usage examples are under the folder [`examples`](https://github.com/rentruewang/bocoel/tree/main/examples). API reference can be found [here](https://bocoel.rentruewang.com/references/overview.html).


## 🥰 Contributing

Contributors wanted! Don't be shy. Feel free to file issues and PRs. For PRs, please follow the guide on [contributing](./CONTRIBUTING.md) and the [code of conduct](./CODE_OF_CONDUCT.md). Openness and inclusiveness are taken very seriously.


## 🗺️ Roadmap: work in progress

- 🪑 Simpler usage. I should provide a high-level wrapper for the entire library s.t. evaluations can be run in one line.
- 📊 Visualization module of the evaluation.
- 🎲 Integration of alternative methods (random, kmedoids...) with Gaussian process.
- 🥨 Integration with more backends such as [VLLM](https://github.com/vllm-project/vllm) and [OpenAI's API](https://github.com/openai/openai-python).
- 🆕 Support for Python 3.12+


## 🏷️ License and Citation

The code is available under [BSD-3 License](./LICENSE.md).

If you find this project helpful in your research, please cite this work at

```
@misc{bocoel2024,
    title = {BoCoEL: Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models},
    url = {https://bocoel.rentruewang.com/research/},
    author = {Wang, RenChu},
    month = {January},
    year = {2024}
}
```

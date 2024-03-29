# Research

_This page is a work in progress. More experiments would be added and the page would be prettier in the future._

--8<-- "README.md"


## Abstract

BoCoEL, short for Bayesian Optimization as a Coverage Tool for Evaluating Large Language Models, represents an innovative approach in the domain of natural language processing (NLP). This framework leverages Bayesian optimization to efficiently evaluate large language models (LLMs) using a significantly reduced yet representative subset of data from extensive corpora. By encoding the data into embeddings and utilizing Bayesian optimization for sample selection, BoCoEL offers a cost-effective and time-efficient alternative for the evaluation of LLMs. This document delineates the methodology, experimentation, and implications of BoCoEL, highlighting its potential to revolutionize the evaluation process in NLP.

## Introduction

The recent proliferation of large language models (LLMs) in NLP has underscored the necessity for efficient evaluation mechanisms. Traditional methods, which involve the assessment of LLMs over vast datasets, are not only time-consuming but also computationally expensive. BoCoEL addresses this challenge by integrating Bayesian optimization into the evaluation process. This approach not only reduces the computational burden but also maintains the integrity and accuracy of the evaluation. The introduction section will further elaborate on the motivation, background, and the specific challenges BoCoEL aims to address in the realm of LLM evaluation.

## Methods

The core methodology of BoCoEL revolves around the use of Bayesian Optimization and embeddings to efficiently evaluate large language models (LLMs). This section provides a detailed mathematical overview of these processes.

### Embedding Process

The embedding process involves transforming corpus entries into a vector space, facilitating efficient manipulation and comparison. Let $\mathcal{D}$ be our dataset containing $N$ entries, $\mathcal{D} = \{d_1, d_2, ..., d_N\}$. Each entry $d_i$ is transformed into an embedding vector $\mathbf{e}_i$ using an embedding function $E$:

$\mathbf{e}_i = E(d_i)$

These embeddings are then used as inputs for the Bayesian Optimization process.

### Bayesian Optimization

Bayesian Optimization (BO) is a strategy for global optimization of black-box functions that are expensive to evaluate. It works well with the LLM evaluation problem, as each evaluation can be computationally intensive.

Let $f: \mathcal{X} \rightarrow \mathbb{R}$ be the expensive black-box function we wish to optimize, where $\mathcal{X}$ is the space of parameters (in our case, the space of embeddings). BO approximates $f$ using a surrogate model, typically a Gaussian Process (GP). A GP is defined by its mean function $m(\mathbf{x})$ and covariance function $k(\mathbf{x}, \mathbf{x'})$, where $\mathbf{x}, \mathbf{x'} \in \mathcal{X}$:

$m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$
$k(\mathbf{x}, \mathbf{x'}) = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x'}) - m(\mathbf{x'}))]$

The GP posterior distribution after observing data $\mathcal{D}_n = \{(\mathbf{x}_1, y_1), ..., (\mathbf{x}_n, y_n)\}$ is also a GP:

$f(\mathbf{x}) | \mathcal{D}_n \sim \mathcal{GP}(\mu_n(\mathbf{x}), \sigma_n^2(\mathbf{x}))$

where

$\mu_n(\mathbf{x}) = k_n(\mathbf{x})^T(K_n + \sigma^2I)^{-1}Y_n$
$\sigma_n^2(\mathbf{x}) = k(\mathbf{x}, \mathbf{x}) - k_n(\mathbf{x})^T(K_n + \sigma^2I)^{-1}k_n(\mathbf{x})$

with $k_n(\mathbf{x}) = [k(\mathbf{x}_1, \mathbf{x}), ..., k(\mathbf{x}_n, \mathbf{x})]^T$, $K_n$ being the covariance matrix formed by applying $k$ to all pairs of points in $\mathcal{D}_n$, and $Y_n = [y_1, ..., y_n]^T$.

An acquisition function $a: \mathcal{X} \rightarrow \mathbb{R}$ is used to determine where to sample next, balancing exploration and exploitation. Common choices for $a$ include Expected Improvement (EI) and Upper Confidence Bound (UCB).

$\text{EI}(\mathbf{x}) = \mathbb{E}[\max(f(\mathbf{x}) - f(\mathbf{x}^+), 0)]$
$\text{UCB}(\mathbf{x}) = \mu_n(\mathbf{x}) + \kappa \sigma_n(\mathbf{x})$

where $\mathbf{x}^+$ is the current best observation, and $\kappa$ is a parameter controlling the exploration-exploitation trade-off.

Here, an entropy search (ES) scheme is used during our evaluation process. This exploration-focused objective ensures that the bayesian process covers as much of the search space with as few samples as possible, which solves the coverage problem (of the embedding space) effectively.

Using this framework, BoCoEL iteratively selects samples from $\mathcal{X}$ (the embedding space) to evaluate the LLM, efficiently optimizing the evaluation process.

## Experiments

TODO: Add plot
TODO: Explain the experiments

The experimental section will present the application of BoCoEL in various scenarios, demonstrating its effectiveness in evaluating different LLMs. Comparative analyses between BoCoEL and traditional evaluation methods will be highlighted, showcasing the efficiency gains in terms of computational resources and time. This section will also include case studies or real-world examples where BoCoEL has been successfully implemented. The results obtained from these experiments will serve to validate the efficacy of the BoCoEL framework in providing accurate evaluations of LLMs using a significantly reduced dataset.

## Authors & Acknowledgement

TODO: Add authors.

See [the thank you page](./thanks.md) for details.

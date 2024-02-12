# Ensemble Experiments

This is the script we used for running our experiments.

As our experiments require in-depth analysis of compoenents, these scripts leverage lower-level APIs that might be difficult (relatively) to get into.

New users are encouraged to use our much-simplified factory methods.

[Documentation here](https://rentruewang.github.io/bocoel/references/factories/)
[Getting started guide](https://github.com/rentruewang/bocoel/tree/main/examples/getting_started)

## Usage

TODO: Add detailed documentation for how to reproduce.

```bash
# Experiments on glue datasets.
python -m examples.ensemble.glue ...

# Experiments on bigbench dataset.
python -m examples.ensemble.bigbench ...

# To plot the results shown in the paper.
python -m examples.ensemble.collect ...
```

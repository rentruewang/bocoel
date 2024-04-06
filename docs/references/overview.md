# Overview

```mermaid
graph TD;
    embedder --> corpus;
    index --> corpus;
    storage --> corpus;
    corpus --> adaptor;
    score --> adaptor;
    adaptor --> optimizer;
    optimizer --> exams;

    index[<a href='https://rentruewang.github.io/bocoel/references/indices'>Index</a>]
    storage[<a href='https://rentruewang.github.io/bocoel/references/storages'>Storage</a>]
    embedder[<a href='https://rentruewang.github.io/bocoel/references/embedders'>Embedder</a>]
    corpus[<a href='https://rentruewang.github.io/bocoel/references/corpora'>Corpus</a>]
    adaptor[<a href='https://rentruewang.github.io/bocoel/references/adaptors'>Adaptor</a>]
    score[<a href='https://rentruewang.github.io/bocoel/references/scores'>Score</a>]
    optimizer[<a href='https://rentruewang.github.io/bocoel/references/optimizers'>Optimizer</a>]
    exams[<a href='https://rentruewang.github.io/bocoel/references/exams'>Exams / Results</a>]
```

# Overview

```{mermaid}
graph TD;
    embedder --> corpus;
    index --> corpus;
    storage --> corpus;
    corpus --> adaptor;
    score --> adaptor;
    adaptor --> optimizer;
    optimizer --> exams;

    index[<a href='./indices.html'>Index</a>]
    storage[<a href='./storages.html'>Storage</a>]
    embedder[<a href='./embedders.html'>Embedder</a>]
    corpus[<a href='./corpora.html'>Corpus</a>]
    adaptor[<a href='./adaptors.html'>Adaptor</a>]
    score[<a href='./scores.html'>Score</a>]
    optimizer[<a href='./optimizers.html'>Optimizer</a>]
    exams[<a href='./exams.html'>Exams / Results</a>]
```

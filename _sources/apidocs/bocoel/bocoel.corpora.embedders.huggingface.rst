:py:mod:`bocoel.corpora.embedders.huggingface`
==============================================

.. py:module:: bocoel.corpora.embedders.huggingface

.. autodoc2-docstring:: bocoel.corpora.embedders.huggingface
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceEmbedder <bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder>`
     - .. autodoc2-docstring:: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder
          :summary:

API
~~~

.. py:class:: HuggingfaceEmbedder(path: str, device: str = 'cpu', batch_size: int = 64, transform: collections.abc.Callable[[typing.Any], torch.Tensor] = lambda output: output.logits)
   :canonical: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder

   Bases: :py:obj:`bocoel.corpora.embedders.interfaces.Embedder`

   .. autodoc2-docstring:: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder.__repr__

   .. py:property:: batch
      :canonical: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder.batch
      :type: int

   .. py:property:: dims
      :canonical: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder.dims
      :type: int

   .. py:method:: _encode(texts: collections.abc.Sequence[str]) -> torch.Tensor
      :canonical: bocoel.corpora.embedders.huggingface.HuggingfaceEmbedder._encode

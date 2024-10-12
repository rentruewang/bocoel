:py:mod:`bocoel.corpora.embedders.sberts`
=========================================

.. py:module:: bocoel.corpora.embedders.sberts

.. autodoc2-docstring:: bocoel.corpora.embedders.sberts
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SbertEmbedder <bocoel.corpora.embedders.sberts.SbertEmbedder>`
     - .. autodoc2-docstring:: bocoel.corpora.embedders.sberts.SbertEmbedder
          :summary:

API
~~~

.. py:class:: SbertEmbedder(model_name: str = 'all-mpnet-base-v2', device: str = 'cpu', batch_size: int = 64)
   :canonical: bocoel.corpora.embedders.sberts.SbertEmbedder

   Bases: :py:obj:`bocoel.corpora.embedders.interfaces.Embedder`

   .. autodoc2-docstring:: bocoel.corpora.embedders.sberts.SbertEmbedder

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.embedders.sberts.SbertEmbedder.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.embedders.sberts.SbertEmbedder.__repr__

   .. py:property:: batch
      :canonical: bocoel.corpora.embedders.sberts.SbertEmbedder.batch
      :type: int

   .. py:property:: dims
      :canonical: bocoel.corpora.embedders.sberts.SbertEmbedder.dims
      :type: int

   .. py:method:: _encode(texts: collections.abc.Sequence[str]) -> torch.Tensor
      :canonical: bocoel.corpora.embedders.sberts.SbertEmbedder._encode

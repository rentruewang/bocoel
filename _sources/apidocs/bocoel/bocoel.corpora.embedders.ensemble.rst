:py:mod:`bocoel.corpora.embedders.ensemble`
===========================================

.. py:module:: bocoel.corpora.embedders.ensemble

.. autodoc2-docstring:: bocoel.corpora.embedders.ensemble
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`EnsembleEmbedder <bocoel.corpora.embedders.ensemble.EnsembleEmbedder>`
     - .. autodoc2-docstring:: bocoel.corpora.embedders.ensemble.EnsembleEmbedder
          :summary:

API
~~~

.. py:class:: EnsembleEmbedder(embedders: collections.abc.Sequence[bocoel.corpora.embedders.interfaces.Embedder], sequential: bool = False)
   :canonical: bocoel.corpora.embedders.ensemble.EnsembleEmbedder

   Bases: :py:obj:`bocoel.corpora.embedders.interfaces.Embedder`

   .. autodoc2-docstring:: bocoel.corpora.embedders.ensemble.EnsembleEmbedder

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.embedders.ensemble.EnsembleEmbedder.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.embedders.ensemble.EnsembleEmbedder.__repr__

   .. py:property:: batch
      :canonical: bocoel.corpora.embedders.ensemble.EnsembleEmbedder.batch
      :type: int

   .. py:property:: dims
      :canonical: bocoel.corpora.embedders.ensemble.EnsembleEmbedder.dims
      :type: int

   .. py:method:: _encode(texts: collections.abc.Sequence[str]) -> torch.Tensor
      :canonical: bocoel.corpora.embedders.ensemble.EnsembleEmbedder._encode

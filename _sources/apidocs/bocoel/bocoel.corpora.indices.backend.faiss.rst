:py:mod:`bocoel.corpora.indices.backend.faiss`
==============================================

.. py:module:: bocoel.corpora.indices.backend.faiss

.. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`FaissIndex <bocoel.corpora.indices.backend.faiss.FaissIndex>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss.FaissIndex
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_faiss <bocoel.corpora.indices.backend.faiss._faiss>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss._faiss
          :summary:

API
~~~

.. py:function:: _faiss()
   :canonical: bocoel.corpora.indices.backend.faiss._faiss

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss._faiss

.. py:class:: FaissIndex(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.Distance, *, normalize: bool = True, index_string: str, cuda: bool = False, batch_size: int = 64)
   :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.Index`

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss.FaissIndex

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss.FaissIndex.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex.__repr__

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex.batch
      :type: int

   .. py:property:: data
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex.data
      :type: numpy.typing.NDArray

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex.distance
      :type: bocoel.corpora.indices.interfaces.Distance

   .. py:property:: dims
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex.dims
      :type: int

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.InternalResult
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex._search

   .. py:method:: _init_index(index_string: str, cuda: bool) -> None
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex._init_index

      .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss.FaissIndex._init_index

   .. py:method:: _faiss_metric(distance: bocoel.corpora.indices.interfaces.Distance) -> typing.Any
      :canonical: bocoel.corpora.indices.backend.faiss.FaissIndex._faiss_metric
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.backend.faiss.FaissIndex._faiss_metric

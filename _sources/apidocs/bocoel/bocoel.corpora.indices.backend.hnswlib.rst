:py:mod:`bocoel.corpora.indices.backend.hnswlib`
================================================

.. py:module:: bocoel.corpora.indices.backend.hnswlib

.. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HnswlibIndex <bocoel.corpora.indices.backend.hnswlib.HnswlibIndex>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_HnswlibDist <bocoel.corpora.indices.backend.hnswlib._HnswlibDist>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib._HnswlibDist
          :summary:

API
~~~

.. py:data:: _HnswlibDist
   :canonical: bocoel.corpora.indices.backend.hnswlib._HnswlibDist
   :value: None

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib._HnswlibDist

.. py:class:: HnswlibIndex(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.Distance, *, normalize: bool = True, threads: int = -1, batch_size: int = 64)
   :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.Index`

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex.__init__

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex.batch
      :type: int

   .. py:property:: data
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex.data
      :type: numpy.typing.NDArray

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex.distance
      :type: bocoel.corpora.indices.interfaces.Distance

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.InternalResult
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex._search

   .. py:method:: _init_index() -> None
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex._init_index

      .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex._init_index

   .. py:method:: _hnswlib_space(distance: bocoel.corpora.indices.interfaces.Distance) -> bocoel.corpora.indices.backend.hnswlib._HnswlibDist
      :canonical: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex._hnswlib_space
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.backend.hnswlib.HnswlibIndex._hnswlib_space

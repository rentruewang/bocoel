:py:mod:`bocoel.corpora.indices.whitening`
==========================================

.. py:module:: bocoel.corpora.indices.whitening

.. autodoc2-docstring:: bocoel.corpora.indices.whitening
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`WhiteningIndex <bocoel.corpora.indices.whitening.WhiteningIndex>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.whitening.WhiteningIndex
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.corpora.indices.whitening.LOGGER>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.whitening.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.corpora.indices.whitening.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.corpora.indices.whitening.LOGGER

.. py:class:: WhiteningIndex(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.Distance, *, reduced: int, whitening_backend: type[bocoel.corpora.indices.interfaces.Index], **backend_kwargs: typing.Any)
   :canonical: bocoel.corpora.indices.whitening.WhiteningIndex

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.Index`

   .. autodoc2-docstring:: bocoel.corpora.indices.whitening.WhiteningIndex

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.whitening.WhiteningIndex.__init__

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex.batch
      :type: int

   .. py:property:: data
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex.data
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.whitening.WhiteningIndex.data

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex.distance
      :type: bocoel.corpora.indices.interfaces.Distance

   .. py:property:: boundary
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex.boundary
      :type: bocoel.corpora.indices.interfaces.Boundary

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.InternalResult
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex._search

   .. py:method:: whiten(embeddings: numpy.typing.NDArray, k: int) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.whitening.WhiteningIndex.whiten
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.whitening.WhiteningIndex.whiten

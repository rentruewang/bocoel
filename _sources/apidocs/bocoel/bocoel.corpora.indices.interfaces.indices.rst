:py:mod:`bocoel.corpora.indices.interfaces.indices`
===================================================

.. py:module:: bocoel.corpora.indices.interfaces.indices

.. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Index <bocoel.corpora.indices.interfaces.indices.Index>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index
          :summary:

API
~~~

.. py:class:: Index(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.distances.Distance, **kwargs: typing.Any)
   :canonical: bocoel.corpora.indices.interfaces.indices.Index

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.__repr__

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.__len__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.__len__

   .. py:method:: __getitem__(idx: int) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.__getitem__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.__getitem__

   .. py:method:: search(query: numpy.typing.ArrayLike, k: int = 1) -> bocoel.corpora.indices.interfaces.results.SearchResultBatch
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.search

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.search

   .. py:method:: in_range(query: numpy.typing.NDArray) -> bool
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.in_range

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.in_range

   .. py:property:: data
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.data
      :abstractmethod:
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.data

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.batch
      :abstractmethod:
      :type: int

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.batch

   .. py:property:: boundary
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.boundary
      :type: bocoel.corpora.indices.interfaces.boundaries.Boundary

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.boundary

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.distance
      :abstractmethod:
      :type: bocoel.corpora.indices.interfaces.distances.Distance

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.distance

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.results.InternalResult
      :canonical: bocoel.corpora.indices.interfaces.indices.Index._search
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index._search

   .. py:property:: dims
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.dims
      :type: int

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.dims

   .. py:property:: lower
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.lower
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.lower

   .. py:property:: upper
      :canonical: bocoel.corpora.indices.interfaces.indices.Index.upper
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.indices.Index.upper

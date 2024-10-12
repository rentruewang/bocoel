:py:mod:`bocoel.corpora.indices.interfaces.results`
===================================================

.. py:module:: bocoel.corpora.indices.interfaces.results

.. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_SearchResult <bocoel.corpora.indices.interfaces.results._SearchResult>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult
          :summary:
   * - :py:obj:`SearchResultBatch <bocoel.corpora.indices.interfaces.results.SearchResultBatch>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResultBatch
          :summary:
   * - :py:obj:`SearchResult <bocoel.corpora.indices.interfaces.results.SearchResult>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResult
          :summary:
   * - :py:obj:`InternalResult <bocoel.corpora.indices.interfaces.results.InternalResult>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.InternalResult
          :summary:

API
~~~

.. py:class:: _SearchResult
   :canonical: bocoel.corpora.indices.interfaces.results._SearchResult

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult

   .. py:attribute:: query
      :canonical: bocoel.corpora.indices.interfaces.results._SearchResult.query
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult.query

   .. py:attribute:: vectors
      :canonical: bocoel.corpora.indices.interfaces.results._SearchResult.vectors
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult.vectors

   .. py:attribute:: distances
      :canonical: bocoel.corpora.indices.interfaces.results._SearchResult.distances
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult.distances

   .. py:attribute:: indices
      :canonical: bocoel.corpora.indices.interfaces.results._SearchResult.indices
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results._SearchResult.indices

.. py:class:: SearchResultBatch
   :canonical: bocoel.corpora.indices.interfaces.results.SearchResultBatch

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.results._SearchResult`

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResultBatch

   .. py:method:: __post_init__() -> None
      :canonical: bocoel.corpora.indices.interfaces.results.SearchResultBatch.__post_init__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResultBatch.__post_init__

.. py:class:: SearchResult
   :canonical: bocoel.corpora.indices.interfaces.results.SearchResult

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.results._SearchResult`

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResult

   .. py:method:: __post_init__() -> None
      :canonical: bocoel.corpora.indices.interfaces.results.SearchResult.__post_init__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.SearchResult.__post_init__

.. py:class:: InternalResult
   :canonical: bocoel.corpora.indices.interfaces.results.InternalResult

   Bases: :py:obj:`typing.NamedTuple`

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.InternalResult

   .. py:attribute:: distances
      :canonical: bocoel.corpora.indices.interfaces.results.InternalResult.distances
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.InternalResult.distances

   .. py:attribute:: indices
      :canonical: bocoel.corpora.indices.interfaces.results.InternalResult.indices
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.results.InternalResult.indices

:py:mod:`bocoel.corpora.indices.utils`
======================================

.. py:module:: bocoel.corpora.indices.utils

.. autodoc2-docstring:: bocoel.corpora.indices.utils
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`validate_embeddings <bocoel.corpora.indices.utils.validate_embeddings>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.utils.validate_embeddings
          :summary:
   * - :py:obj:`normalize <bocoel.corpora.indices.utils.normalize>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.utils.normalize
          :summary:
   * - :py:obj:`boundaries <bocoel.corpora.indices.utils.boundaries>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.utils.boundaries
          :summary:
   * - :py:obj:`split_search_result_batch <bocoel.corpora.indices.utils.split_search_result_batch>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.utils.split_search_result_batch
          :summary:
   * - :py:obj:`join_search_results <bocoel.corpora.indices.utils.join_search_results>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.utils.join_search_results
          :summary:

API
~~~

.. py:function:: validate_embeddings(embeddings: numpy.typing.NDArray, /, allowed_ndims: int | list[int] = 2) -> None
   :canonical: bocoel.corpora.indices.utils.validate_embeddings

   .. autodoc2-docstring:: bocoel.corpora.indices.utils.validate_embeddings

.. py:function:: normalize(embeddings: numpy.typing.ArrayLike, /, p: int = 2) -> numpy.typing.NDArray
   :canonical: bocoel.corpora.indices.utils.normalize

   .. autodoc2-docstring:: bocoel.corpora.indices.utils.normalize

.. py:function:: boundaries(embeddings: numpy.typing.NDArray, /) -> bocoel.corpora.indices.interfaces.Boundary
   :canonical: bocoel.corpora.indices.utils.boundaries

   .. autodoc2-docstring:: bocoel.corpora.indices.utils.boundaries

.. py:function:: split_search_result_batch(srb: bocoel.corpora.indices.interfaces.SearchResultBatch, /) -> list[bocoel.corpora.indices.interfaces.SearchResult]
   :canonical: bocoel.corpora.indices.utils.split_search_result_batch

   .. autodoc2-docstring:: bocoel.corpora.indices.utils.split_search_result_batch

.. py:function:: join_search_results(srs: collections.abc.Iterable[bocoel.corpora.indices.interfaces.SearchResult], /) -> bocoel.corpora.indices.interfaces.SearchResultBatch
   :canonical: bocoel.corpora.indices.utils.join_search_results

   .. autodoc2-docstring:: bocoel.corpora.indices.utils.join_search_results

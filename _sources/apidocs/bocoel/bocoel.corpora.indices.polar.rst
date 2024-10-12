:py:mod:`bocoel.corpora.indices.polar`
======================================

.. py:module:: bocoel.corpora.indices.polar

.. autodoc2-docstring:: bocoel.corpora.indices.polar
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PolarIndex <bocoel.corpora.indices.polar.PolarIndex>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.corpora.indices.polar.LOGGER>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.polar.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.corpora.indices.polar.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.corpora.indices.polar.LOGGER

.. py:class:: PolarIndex(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.Distance, *, polar_backend: type[bocoel.corpora.indices.interfaces.Index], **backend_kwargs: typing.Any)
   :canonical: bocoel.corpora.indices.polar.PolarIndex

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.Index`

   .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex.__init__

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.InternalResult
      :canonical: bocoel.corpora.indices.polar.PolarIndex._search

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.polar.PolarIndex.batch
      :type: int

   .. py:property:: data
      :canonical: bocoel.corpora.indices.polar.PolarIndex.data
      :type: numpy.typing.NDArray

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.polar.PolarIndex.distance
      :type: bocoel.corpora.indices.interfaces.Distance

   .. py:property:: boundary
      :canonical: bocoel.corpora.indices.polar.PolarIndex.boundary
      :type: bocoel.corpora.indices.interfaces.Boundary

   .. py:method:: _polar_boundary(dims: int) -> bocoel.corpora.indices.interfaces.Boundary
      :canonical: bocoel.corpora.indices.polar.PolarIndex._polar_boundary

      .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex._polar_boundary

   .. py:method:: _polar_coordinates() -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.polar.PolarIndex._polar_coordinates

      .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex._polar_coordinates

   .. py:method:: polar_to_spatial(r: numpy.typing.ArrayLike, theta: numpy.typing.ArrayLike) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.polar.PolarIndex.polar_to_spatial
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex.polar_to_spatial

   .. py:method:: spatial_to_polar(x: numpy.typing.ArrayLike) -> tuple[numpy.typing.NDArray, numpy.typing.NDArray]
      :canonical: bocoel.corpora.indices.polar.PolarIndex.spatial_to_polar
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.polar.PolarIndex.spatial_to_polar

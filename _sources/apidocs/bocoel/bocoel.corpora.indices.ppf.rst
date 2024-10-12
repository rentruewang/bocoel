:py:mod:`bocoel.corpora.indices.ppf`
====================================

.. py:module:: bocoel.corpora.indices.ppf

.. autodoc2-docstring:: bocoel.corpora.indices.ppf
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Distribution <bocoel.corpora.indices.ppf.Distribution>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution
          :summary:
   * - :py:obj:`InverseCDFIndex <bocoel.corpora.indices.ppf.InverseCDFIndex>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.ppf.InverseCDFIndex
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.corpora.indices.ppf.LOGGER>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.ppf.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.corpora.indices.ppf.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.corpora.indices.ppf.LOGGER

.. py:class:: Distribution()
   :canonical: bocoel.corpora.indices.ppf.Distribution

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution.__init__

   .. py:attribute:: NORMAL
      :canonical: bocoel.corpora.indices.ppf.Distribution.NORMAL
      :value: 'NORMAL'

      .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution.NORMAL

   .. py:attribute:: UNIFORM
      :canonical: bocoel.corpora.indices.ppf.Distribution.UNIFORM
      :value: 'UNIFORM'

      .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution.UNIFORM

   .. py:property:: cdf
      :canonical: bocoel.corpora.indices.ppf.Distribution.cdf
      :type: collections.abc.Callable[[numpy.typing.ArrayLike], numpy.typing.NDArray]

      .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution.cdf

   .. py:property:: ppf
      :canonical: bocoel.corpora.indices.ppf.Distribution.ppf
      :type: collections.abc.Callable[[numpy.typing.ArrayLike], numpy.typing.NDArray]

      .. autodoc2-docstring:: bocoel.corpora.indices.ppf.Distribution.ppf

.. py:class:: InverseCDFIndex(embeddings: numpy.typing.NDArray, distance: str | bocoel.corpora.indices.interfaces.Distance, *, distribution: str | bocoel.corpora.indices.ppf.Distribution = Distribution.NORMAL, inverse_cdf_backend: type[bocoel.corpora.indices.interfaces.Index], **backend_kwargs: typing.Any)
   :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex

   Bases: :py:obj:`bocoel.corpora.indices.interfaces.Index`

   .. autodoc2-docstring:: bocoel.corpora.indices.ppf.InverseCDFIndex

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.indices.ppf.InverseCDFIndex.__init__

   .. py:method:: _search(query: numpy.typing.NDArray, k: int = 1) -> bocoel.corpora.indices.interfaces.InternalResult
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex._search

   .. py:property:: batch
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex.batch
      :type: int

   .. py:property:: data
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex.data
      :type: numpy.typing.NDArray

   .. py:property:: distance
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex.distance
      :type: bocoel.corpora.indices.interfaces.Distance

   .. py:property:: dims
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex.dims
      :type: int

   .. py:property:: boundary
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex.boundary
      :type: bocoel.corpora.indices.interfaces.Boundary

   .. py:method:: _cdf() -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.ppf.InverseCDFIndex._cdf

      .. autodoc2-docstring:: bocoel.corpora.indices.ppf.InverseCDFIndex._cdf

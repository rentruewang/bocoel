:py:mod:`bocoel.corpora.indices.interfaces.boundaries`
======================================================

.. py:module:: bocoel.corpora.indices.interfaces.boundaries

.. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Boundary <bocoel.corpora.indices.interfaces.boundaries.Boundary>`
     - .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary
          :summary:

API
~~~

.. py:class:: Boundary
   :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary

   .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary

   .. py:attribute:: bounds
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.bounds
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.bounds

   .. py:method:: __post_init__() -> None
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.__post_init__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.__post_init__

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.__len__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.__len__

   .. py:method:: __getitem__(idx: int, /) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.__getitem__

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.__getitem__

   .. py:property:: dims
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.dims
      :type: int

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.dims

   .. py:property:: lower
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.lower
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.lower

   .. py:property:: upper
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.upper
      :type: numpy.typing.NDArray

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.upper

   .. py:method:: fixed(lower: float, upper: float, dims: int) -> typing_extensions.Self
      :canonical: bocoel.corpora.indices.interfaces.boundaries.Boundary.fixed
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.indices.interfaces.boundaries.Boundary.fixed

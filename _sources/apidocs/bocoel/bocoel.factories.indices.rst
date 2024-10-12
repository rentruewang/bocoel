:py:mod:`bocoel.factories.indices`
==================================

.. py:module:: bocoel.factories.indices

.. autodoc2-docstring:: bocoel.factories.indices
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`IndexName <bocoel.factories.indices.IndexName>`
     - .. autodoc2-docstring:: bocoel.factories.indices.IndexName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`index_class <bocoel.factories.indices.index_class>`
     - .. autodoc2-docstring:: bocoel.factories.indices.index_class
          :summary:
   * - :py:obj:`index_set_backends <bocoel.factories.indices.index_set_backends>`
     - .. autodoc2-docstring:: bocoel.factories.indices.index_set_backends
          :summary:

API
~~~

.. py:class:: IndexName()
   :canonical: bocoel.factories.indices.IndexName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.indices.IndexName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.indices.IndexName.__init__

   .. py:attribute:: FAISS
      :canonical: bocoel.factories.indices.IndexName.FAISS
      :value: 'FAISS'

      .. autodoc2-docstring:: bocoel.factories.indices.IndexName.FAISS

   .. py:attribute:: HNSWLIB
      :canonical: bocoel.factories.indices.IndexName.HNSWLIB
      :value: 'HNSWLIB'

      .. autodoc2-docstring:: bocoel.factories.indices.IndexName.HNSWLIB

   .. py:attribute:: POLAR
      :canonical: bocoel.factories.indices.IndexName.POLAR
      :value: 'POLAR'

      .. autodoc2-docstring:: bocoel.factories.indices.IndexName.POLAR

   .. py:attribute:: WHITENING
      :canonical: bocoel.factories.indices.IndexName.WHITENING
      :value: 'WHITENING'

      .. autodoc2-docstring:: bocoel.factories.indices.IndexName.WHITENING

.. py:function:: index_class(name: str | bocoel.factories.indices.IndexName, /) -> type[bocoel.Index]
   :canonical: bocoel.factories.indices.index_class

   .. autodoc2-docstring:: bocoel.factories.indices.index_class

.. py:function:: index_set_backends(kwargs: dict[str, typing.Any], /) -> dict[str, typing.Any]
   :canonical: bocoel.factories.indices.index_set_backends

   .. autodoc2-docstring:: bocoel.factories.indices.index_set_backends

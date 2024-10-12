:py:mod:`bocoel.corpora.storages.concat`
========================================

.. py:module:: bocoel.corpora.storages.concat

.. autodoc2-docstring:: bocoel.corpora.storages.concat
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ConcatStorage <bocoel.corpora.storages.concat.ConcatStorage>`
     - .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.corpora.storages.concat.LOGGER>`
     - .. autodoc2-docstring:: bocoel.corpora.storages.concat.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.corpora.storages.concat.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.corpora.storages.concat.LOGGER

.. py:class:: ConcatStorage(storages: collections.abc.Sequence[bocoel.corpora.storages.interfaces.Storage], /)
   :canonical: bocoel.corpora.storages.concat.ConcatStorage

   Bases: :py:obj:`bocoel.corpora.storages.interfaces.Storage`

   .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage.__init__

   .. py:method:: __repr___() -> str
      :canonical: bocoel.corpora.storages.concat.ConcatStorage.__repr___

      .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage.__repr___

   .. py:method:: keys() -> collections.abc.Collection[str]
      :canonical: bocoel.corpora.storages.concat.ConcatStorage.keys

      .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage.keys

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.storages.concat.ConcatStorage.__len__

   .. py:method:: _getitem(idx: int) -> collections.abc.Mapping[str, typing.Any]
      :canonical: bocoel.corpora.storages.concat.ConcatStorage._getitem

   .. py:method:: join(storages: collections.abc.Iterable[bocoel.corpora.storages.interfaces.Storage], /) -> bocoel.corpora.storages.interfaces.Storage
      :canonical: bocoel.corpora.storages.concat.ConcatStorage.join
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.concat.ConcatStorage.join

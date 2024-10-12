:py:mod:`bocoel.corpora.storages.datasets`
==========================================

.. py:module:: bocoel.corpora.storages.datasets

.. autodoc2-docstring:: bocoel.corpora.storages.datasets
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`DatasetsStorage <bocoel.corpora.storages.datasets.DatasetsStorage>`
     - .. autodoc2-docstring:: bocoel.corpora.storages.datasets.DatasetsStorage
          :summary:

API
~~~

.. py:class:: DatasetsStorage(path: str, name: str | None = None, split: str | None = None)
   :canonical: bocoel.corpora.storages.datasets.DatasetsStorage

   Bases: :py:obj:`bocoel.corpora.storages.interfaces.Storage`

   .. autodoc2-docstring:: bocoel.corpora.storages.datasets.DatasetsStorage

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.storages.datasets.DatasetsStorage.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.storages.datasets.DatasetsStorage.__repr__

   .. py:method:: keys() -> collections.abc.Collection[str]
      :canonical: bocoel.corpora.storages.datasets.DatasetsStorage.keys

      .. autodoc2-docstring:: bocoel.corpora.storages.datasets.DatasetsStorage.keys

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.storages.datasets.DatasetsStorage.__len__

   .. py:method:: _getitem(idx: int) -> collections.abc.Mapping[str, typing.Any]
      :canonical: bocoel.corpora.storages.datasets.DatasetsStorage._getitem

:py:mod:`bocoel.corpora.storages.interfaces`
============================================

.. py:module:: bocoel.corpora.storages.interfaces

.. autodoc2-docstring:: bocoel.corpora.storages.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Storage <bocoel.corpora.storages.interfaces.Storage>`
     - .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage
          :summary:

API
~~~

.. py:class:: Storage
   :canonical: bocoel.corpora.storages.interfaces.Storage

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.storages.interfaces.Storage.__repr__

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.storages.interfaces.Storage.__len__
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage.__len__

   .. py:method:: __getitem__(idx: int | slice | collections.abc.Sequence[int]) -> collections.abc.Mapping[str, typing.Any] | collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]
      :canonical: bocoel.corpora.storages.interfaces.Storage.__getitem__

      .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage.__getitem__

   .. py:method:: _getitem(idx: int) -> collections.abc.Mapping[str, typing.Any]
      :canonical: bocoel.corpora.storages.interfaces.Storage._getitem
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage._getitem

   .. py:method:: keys() -> collections.abc.Collection[str]
      :canonical: bocoel.corpora.storages.interfaces.Storage.keys
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage.keys

   .. py:method:: collate(mappings: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]) -> collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]
      :canonical: bocoel.corpora.storages.interfaces.Storage.collate
      :staticmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.interfaces.Storage.collate

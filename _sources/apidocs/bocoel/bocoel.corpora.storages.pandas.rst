:py:mod:`bocoel.corpora.storages.pandas`
========================================

.. py:module:: bocoel.corpora.storages.pandas

.. autodoc2-docstring:: bocoel.corpora.storages.pandas
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PandasStorage <bocoel.corpora.storages.pandas.PandasStorage>`
     - .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage
          :summary:

API
~~~

.. py:class:: PandasStorage(df: pandas.DataFrame, /)
   :canonical: bocoel.corpora.storages.pandas.PandasStorage

   Bases: :py:obj:`bocoel.corpora.storages.interfaces.Storage`

   .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage.__init__

   .. py:method:: keys() -> collections.abc.Collection[str]
      :canonical: bocoel.corpora.storages.pandas.PandasStorage.keys

      .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage.keys

   .. py:method:: __len__() -> int
      :canonical: bocoel.corpora.storages.pandas.PandasStorage.__len__

   .. py:method:: _getitem(idx: int) -> collections.abc.Mapping[str, typing.Any]
      :canonical: bocoel.corpora.storages.pandas.PandasStorage._getitem

   .. py:method:: from_jsonl_file(path: str | pathlib.Path, /) -> typing_extensions.Self
      :canonical: bocoel.corpora.storages.pandas.PandasStorage.from_jsonl_file
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage.from_jsonl_file

   .. py:method:: from_jsonl(data: collections.abc.Sequence[collections.abc.Mapping[str, str]], /) -> typing_extensions.Self
      :canonical: bocoel.corpora.storages.pandas.PandasStorage.from_jsonl
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.storages.pandas.PandasStorage.from_jsonl

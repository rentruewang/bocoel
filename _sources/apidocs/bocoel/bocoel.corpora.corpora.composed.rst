:py:mod:`bocoel.corpora.corpora.composed`
=========================================

.. py:module:: bocoel.corpora.corpora.composed

.. autodoc2-docstring:: bocoel.corpora.corpora.composed
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ComposedCorpus <bocoel.corpora.corpora.composed.ComposedCorpus>`
     - .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus
          :summary:

API
~~~

.. py:class:: ComposedCorpus
   :canonical: bocoel.corpora.corpora.composed.ComposedCorpus

   Bases: :py:obj:`bocoel.corpora.corpora.interfaces.Corpus`

   .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus

   .. py:attribute:: index
      :canonical: bocoel.corpora.corpora.composed.ComposedCorpus.index
      :type: bocoel.corpora.indices.Index
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus.index

   .. py:attribute:: storage
      :canonical: bocoel.corpora.corpora.composed.ComposedCorpus.storage
      :type: bocoel.corpora.storages.Storage
      :value: None

      .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus.storage

   .. py:method:: index_storage(storage: bocoel.corpora.storages.Storage, embedder: bocoel.corpora.embedders.Embedder, keys: collections.abc.Sequence[str], index_backend: type[bocoel.corpora.indices.Index], concat: collections.abc.Callable[[collections.abc.Iterable[typing.Any]], str] = ' [SEP] '.join, **index_kwargs: typing.Any) -> typing_extensions.Self
      :canonical: bocoel.corpora.corpora.composed.ComposedCorpus.index_storage
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus.index_storage

   .. py:method:: index_mapped(storage: bocoel.corpora.storages.Storage, embedder: bocoel.corpora.embedders.Embedder, transform: collections.abc.Callable[[collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]], collections.abc.Sequence[str]], index_backend: type[bocoel.corpora.indices.Index], **index_kwargs: typing.Any) -> typing_extensions.Self
      :canonical: bocoel.corpora.corpora.composed.ComposedCorpus.index_mapped
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus.index_mapped

   .. py:method:: index_embeddings(storage: bocoel.corpora.storages.Storage, embeddings: numpy.typing.NDArray, index_backend: type[bocoel.corpora.indices.Index], **index_kwargs: typing.Any) -> typing_extensions.Self
      :canonical: bocoel.corpora.corpora.composed.ComposedCorpus.index_embeddings
      :classmethod:

      .. autodoc2-docstring:: bocoel.corpora.corpora.composed.ComposedCorpus.index_embeddings

:py:mod:`bocoel.corpora.embedders.interfaces`
=============================================

.. py:module:: bocoel.corpora.embedders.interfaces

.. autodoc2-docstring:: bocoel.corpora.embedders.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Embedder <bocoel.corpora.embedders.interfaces.Embedder>`
     - .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.corpora.embedders.interfaces.LOGGER>`
     - .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.corpora.embedders.interfaces.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.LOGGER

.. py:class:: Embedder
   :canonical: bocoel.corpora.embedders.interfaces.Embedder

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder

   .. py:method:: __repr__() -> str
      :canonical: bocoel.corpora.embedders.interfaces.Embedder.__repr__

   .. py:method:: encode_storage(storage: bocoel.corpora.storages.Storage, /, transform: collections.abc.Callable[[collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]], collections.abc.Sequence[str]]) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.embedders.interfaces.Embedder.encode_storage

      .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder.encode_storage

   .. py:method:: encode(text: collections.abc.Sequence[str], /) -> numpy.typing.NDArray
      :canonical: bocoel.corpora.embedders.interfaces.Embedder.encode

      .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder.encode

   .. py:property:: batch
      :canonical: bocoel.corpora.embedders.interfaces.Embedder.batch
      :abstractmethod:
      :type: int

      .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder.batch

   .. py:property:: dims
      :canonical: bocoel.corpora.embedders.interfaces.Embedder.dims
      :abstractmethod:
      :type: int

      .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder.dims

   .. py:method:: _encode(texts: collections.abc.Sequence[str], /) -> torch.Tensor
      :canonical: bocoel.corpora.embedders.interfaces.Embedder._encode
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.corpora.embedders.interfaces.Embedder._encode

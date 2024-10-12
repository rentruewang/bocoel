:py:mod:`bocoel.models.adaptors.interfaces.adaptors`
====================================================

.. py:module:: bocoel.models.adaptors.interfaces.adaptors

.. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Adaptor <bocoel.models.adaptors.interfaces.adaptors.Adaptor>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.Adaptor
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.models.adaptors.interfaces.adaptors.LOGGER>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.models.adaptors.interfaces.adaptors.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.LOGGER

.. py:class:: Adaptor
   :canonical: bocoel.models.adaptors.interfaces.adaptors.Adaptor

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.Adaptor

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.adaptors.interfaces.adaptors.Adaptor.__repr__

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Sequence[float] | numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.interfaces.adaptors.Adaptor.evaluate
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.Adaptor.evaluate

   .. py:method:: on_storage(storage: bocoel.corpora.Storage, indices: numpy.typing.ArrayLike) -> numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.interfaces.adaptors.Adaptor.on_storage

      .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.Adaptor.on_storage

   .. py:method:: on_corpus(corpus: bocoel.corpora.Corpus, indices: numpy.typing.ArrayLike) -> numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.interfaces.adaptors.Adaptor.on_corpus

      .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.adaptors.Adaptor.on_corpus

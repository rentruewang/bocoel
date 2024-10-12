:py:mod:`bocoel.factories.corpora`
==================================

.. py:module:: bocoel.factories.corpora

.. autodoc2-docstring:: bocoel.factories.corpora
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CorpusName <bocoel.factories.corpora.CorpusName>`
     - .. autodoc2-docstring:: bocoel.factories.corpora.CorpusName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`corpus <bocoel.factories.corpora.corpus>`
     - .. autodoc2-docstring:: bocoel.factories.corpora.corpus
          :summary:

API
~~~

.. py:class:: CorpusName()
   :canonical: bocoel.factories.corpora.CorpusName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.corpora.CorpusName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.corpora.CorpusName.__init__

   .. py:attribute:: COMPOSED
      :canonical: bocoel.factories.corpora.CorpusName.COMPOSED
      :value: 'COMPOSED'

      .. autodoc2-docstring:: bocoel.factories.corpora.CorpusName.COMPOSED

.. py:function:: corpus(name: str | bocoel.factories.corpora.CorpusName = CorpusName.COMPOSED, /, *, storage: bocoel.Storage, embedder: bocoel.Embedder, keys: collections.abc.Sequence[str], index_name: str | bocoel.factories.indices.IndexName, **index_kwargs: typing.Any) -> bocoel.Corpus
   :canonical: bocoel.factories.corpora.corpus

   .. autodoc2-docstring:: bocoel.factories.corpora.corpus

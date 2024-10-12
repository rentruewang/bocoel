:py:mod:`bocoel.core.optim.corpora`
===================================

.. py:module:: bocoel.core.optim.corpora

.. autodoc2-docstring:: bocoel.core.optim.corpora
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`CorpusEvaluator <bocoel.core.optim.corpora.CorpusEvaluator>`
     - .. autodoc2-docstring:: bocoel.core.optim.corpora.CorpusEvaluator
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.optim.corpora.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.optim.corpora.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.optim.corpora.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.optim.corpora.LOGGER

.. py:class:: CorpusEvaluator(corpus: bocoel.corpora.Corpus, adaptor: bocoel.models.Adaptor)
   :canonical: bocoel.core.optim.corpora.CorpusEvaluator

   Bases: :py:obj:`bocoel.core.optim.interfaces.IndexEvaluator`

   .. autodoc2-docstring:: bocoel.core.optim.corpora.CorpusEvaluator

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.corpora.CorpusEvaluator.__init__

   .. py:method:: __call__(idx: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray
      :canonical: bocoel.core.optim.corpora.CorpusEvaluator.__call__

      .. autodoc2-docstring:: bocoel.core.optim.corpora.CorpusEvaluator.__call__

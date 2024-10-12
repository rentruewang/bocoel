:py:mod:`bocoel.models.adaptors.bigbench.matching`
==================================================

.. py:module:: bocoel.models.adaptors.bigbench.matching

.. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BigBenchMatchType <bocoel.models.adaptors.bigbench.matching.BigBenchMatchType>`
     -
   * - :py:obj:`BigBenchQuestionAnswer <bocoel.models.adaptors.bigbench.matching.BigBenchQuestionAnswer>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.models.adaptors.bigbench.matching.LOGGER>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.models.adaptors.bigbench.matching.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.LOGGER

.. py:class:: BigBenchMatchType()
   :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. py:attribute:: EXACT
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.EXACT
      :value: 'EXACT'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.EXACT

   .. py:attribute:: NLTK_BLEU
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.NLTK_BLEU
      :value: 'NLTK_BLEU'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.NLTK_BLEU

   .. py:attribute:: SACRE_BLEU
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.SACRE_BLEU
      :value: 'SACRE_BLEU'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.SACRE_BLEU

   .. py:attribute:: ROUGE_1
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_1
      :value: 'ROUGE_1'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_1

   .. py:attribute:: ROUGE_2
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_2
      :value: 'ROUGE_2'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_2

   .. py:attribute:: ROUGE_L
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_L
      :value: 'ROUGE_L'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_L

   .. py:attribute:: ROUGE_SCORE_1
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_1
      :value: 'ROUGE_SCORE_1'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_1

   .. py:attribute:: ROUGE_SCORE_2
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_2
      :value: 'ROUGE_SCORE_2'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_2

   .. py:attribute:: ROUGE_SCORE_L
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_L
      :value: 'ROUGE_SCORE_L'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.ROUGE_SCORE_L

   .. py:property:: score
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.score
      :type: bocoel.models.scores.Score

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.matching.BigBenchMatchType.score

.. py:class:: BigBenchQuestionAnswer(lm: bocoel.models.lms.GenerativeModel, inputs: str = 'inputs', targets: str = 'targets', matching_type: str | bocoel.models.adaptors.bigbench.matching.BigBenchMatchType = BigBenchMatchType.EXACT)
   :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchQuestionAnswer

   Bases: :py:obj:`bocoel.models.adaptors.bigbench.interfaces.BigBenchAdaptor`

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchQuestionAnswer.__repr__

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Sequence[float] | numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.bigbench.matching.BigBenchQuestionAnswer.evaluate

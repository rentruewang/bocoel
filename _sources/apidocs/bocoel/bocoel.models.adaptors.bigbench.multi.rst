:py:mod:`bocoel.models.adaptors.bigbench.multi`
===============================================

.. py:module:: bocoel.models.adaptors.bigbench.multi

.. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BigBenchChoiceType <bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType>`
     -
   * - :py:obj:`BigBenchMultipleChoice <bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.models.adaptors.bigbench.multi.LOGGER>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.models.adaptors.bigbench.multi.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.LOGGER

.. py:class:: BigBenchChoiceType()
   :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. py:attribute:: SUM_OF_SCORES
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.SUM_OF_SCORES
      :value: 'SUM_OF_SCORES'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.SUM_OF_SCORES

   .. py:attribute:: LIST_OF_ANSWERS
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.LIST_OF_ANSWERS
      :value: 'LIST_OF_ANSWERS'

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.LIST_OF_ANSWERS

   .. py:property:: score
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.score
      :type: bocoel.models.scores.Score

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType.score

.. py:class:: BigBenchMultipleChoice(lm: bocoel.models.lms.ClassifierModel, inputs: str = 'inputs', multiple_choice_targets: str = 'multiple_choice_targets', multiple_choice_scores: str = 'multiple_choice_scores', choice_type: str | bocoel.models.adaptors.bigbench.multi.BigBenchChoiceType = BigBenchChoiceType.SUM_OF_SCORES)
   :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice

   Bases: :py:obj:`bocoel.models.adaptors.bigbench.interfaces.BigBenchAdaptor`

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice.__repr__

   .. py:method:: evaluate(data: collections.abc.Mapping[str, typing.Any]) -> collections.abc.Sequence[float] | numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice.evaluate

   .. py:method:: numeric_choices(question: str, choices: collections.abc.Sequence[str]) -> str
      :canonical: bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice.numeric_choices
      :staticmethod:

      .. autodoc2-docstring:: bocoel.models.adaptors.bigbench.multi.BigBenchMultipleChoice.numeric_choices

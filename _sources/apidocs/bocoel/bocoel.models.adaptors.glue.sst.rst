:py:mod:`bocoel.models.adaptors.glue.sst`
=========================================

.. py:module:: bocoel.models.adaptors.glue.sst

.. autodoc2-docstring:: bocoel.models.adaptors.glue.sst
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Sst2QuestionAnswer <bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer
          :summary:

API
~~~

.. py:class:: Sst2QuestionAnswer(lm: bocoel.models.lms.ClassifierModel, sentence: str = 'sentence', label: str = 'label', choices: collections.abc.Sequence[str] = ('negative', 'positive'))
   :canonical: bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer

   Bases: :py:obj:`bocoel.models.adaptors.interfaces.Adaptor`

   .. autodoc2-docstring:: bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer.__init__

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Sequence[float] | numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.glue.sst.Sst2QuestionAnswer.evaluate

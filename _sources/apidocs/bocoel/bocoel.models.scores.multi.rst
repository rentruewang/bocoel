:py:mod:`bocoel.models.scores.multi`
====================================

.. py:module:: bocoel.models.scores.multi

.. autodoc2-docstring:: bocoel.models.scores.multi
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`OneHotChoiceAccuracy <bocoel.models.scores.multi.OneHotChoiceAccuracy>`
     -
   * - :py:obj:`MultiChoiceAccuracy <bocoel.models.scores.multi.MultiChoiceAccuracy>`
     -

API
~~~

.. py:class:: OneHotChoiceAccuracy
   :canonical: bocoel.models.scores.multi.OneHotChoiceAccuracy

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: int, references: list[float]) -> float
      :canonical: bocoel.models.scores.multi.OneHotChoiceAccuracy.__call__

.. py:class:: MultiChoiceAccuracy
   :canonical: bocoel.models.scores.multi.MultiChoiceAccuracy

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: int, references: list[int]) -> float
      :canonical: bocoel.models.scores.multi.MultiChoiceAccuracy.__call__

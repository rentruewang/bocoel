:py:mod:`bocoel.models.scores.rouge`
====================================

.. py:module:: bocoel.models.scores.rouge

.. autodoc2-docstring:: bocoel.models.scores.rouge
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RougeScore <bocoel.models.scores.rouge.RougeScore>`
     -
   * - :py:obj:`RougeScore2 <bocoel.models.scores.rouge.RougeScore2>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_RougeScoreType <bocoel.models.scores.rouge._RougeScoreType>`
     - .. autodoc2-docstring:: bocoel.models.scores.rouge._RougeScoreType
          :summary:
   * - :py:obj:`_RougeScore2Type <bocoel.models.scores.rouge._RougeScore2Type>`
     - .. autodoc2-docstring:: bocoel.models.scores.rouge._RougeScore2Type
          :summary:

API
~~~

.. py:data:: _RougeScoreType
   :canonical: bocoel.models.scores.rouge._RougeScoreType
   :value: None

   .. autodoc2-docstring:: bocoel.models.scores.rouge._RougeScoreType

.. py:class:: RougeScore(metric: bocoel.models.scores.rouge._RougeScoreType)
   :canonical: bocoel.models.scores.rouge.RougeScore

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: str, references: list[str]) -> float
      :canonical: bocoel.models.scores.rouge.RougeScore.__call__

.. py:data:: _RougeScore2Type
   :canonical: bocoel.models.scores.rouge._RougeScore2Type
   :value: None

   .. autodoc2-docstring:: bocoel.models.scores.rouge._RougeScore2Type

.. py:class:: RougeScore2(typ: bocoel.models.scores.rouge._RougeScore2Type)
   :canonical: bocoel.models.scores.rouge.RougeScore2

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: typing.Any, references: list[str]) -> float
      :canonical: bocoel.models.scores.rouge.RougeScore2.__call__

:py:mod:`bocoel.models.scores.bleu`
===================================

.. py:module:: bocoel.models.scores.bleu

.. autodoc2-docstring:: bocoel.models.scores.bleu
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`NltkBleuScore <bocoel.models.scores.bleu.NltkBleuScore>`
     -
   * - :py:obj:`SacreBleuScore <bocoel.models.scores.bleu.SacreBleuScore>`
     -

API
~~~

.. py:class:: NltkBleuScore
   :canonical: bocoel.models.scores.bleu.NltkBleuScore

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: str, references: list[str]) -> float
      :canonical: bocoel.models.scores.bleu.NltkBleuScore.__call__

.. py:class:: SacreBleuScore()
   :canonical: bocoel.models.scores.bleu.SacreBleuScore

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: str, references: list[str]) -> float
      :canonical: bocoel.models.scores.bleu.SacreBleuScore.__call__

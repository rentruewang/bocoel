:py:mod:`bocoel.models.scores.exact`
====================================

.. py:module:: bocoel.models.scores.exact

.. autodoc2-docstring:: bocoel.models.scores.exact
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ExactMatch <bocoel.models.scores.exact.ExactMatch>`
     -

API
~~~

.. py:class:: ExactMatch
   :canonical: bocoel.models.scores.exact.ExactMatch

   Bases: :py:obj:`bocoel.models.scores.interfaces.Score`

   .. py:method:: __call__(target: str, references: list[str]) -> float
      :canonical: bocoel.models.scores.exact.ExactMatch.__call__

   .. py:method:: _clean(string: str) -> str
      :canonical: bocoel.models.scores.exact.ExactMatch._clean
      :staticmethod:

      .. autodoc2-docstring:: bocoel.models.scores.exact.ExactMatch._clean

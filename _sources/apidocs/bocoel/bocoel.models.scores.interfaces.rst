:py:mod:`bocoel.models.scores.interfaces`
=========================================

.. py:module:: bocoel.models.scores.interfaces

.. autodoc2-docstring:: bocoel.models.scores.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Score <bocoel.models.scores.interfaces.Score>`
     -

API
~~~

.. py:class:: Score
   :canonical: bocoel.models.scores.interfaces.Score

   Bases: :py:obj:`typing.Protocol`

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.scores.interfaces.Score.__repr__

   .. py:method:: __call__(target: typing.Any, references: list[typing.Any]) -> float
      :canonical: bocoel.models.scores.interfaces.Score.__call__
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.models.scores.interfaces.Score.__call__

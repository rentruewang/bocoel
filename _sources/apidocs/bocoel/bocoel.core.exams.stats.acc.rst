:py:mod:`bocoel.core.exams.stats.acc`
=====================================

.. py:module:: bocoel.core.exams.stats.acc

.. autodoc2-docstring:: bocoel.core.exams.stats.acc
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AccType <bocoel.core.exams.stats.acc.AccType>`
     - .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType
          :summary:
   * - :py:obj:`Accumulation <bocoel.core.exams.stats.acc.Accumulation>`
     - .. autodoc2-docstring:: bocoel.core.exams.stats.acc.Accumulation
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_check_dim <bocoel.core.exams.stats.acc._check_dim>`
     - .. autodoc2-docstring:: bocoel.core.exams.stats.acc._check_dim
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.exams.stats.acc.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.exams.stats.acc.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.exams.stats.acc.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc.LOGGER

.. py:class:: AccType()
   :canonical: bocoel.core.exams.stats.acc.AccType

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType.__init__

   .. py:attribute:: MIN
      :canonical: bocoel.core.exams.stats.acc.AccType.MIN
      :value: 'MINIMUM'

      .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType.MIN

   .. py:attribute:: MAX
      :canonical: bocoel.core.exams.stats.acc.AccType.MAX
      :value: 'MAXIMUM'

      .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType.MAX

   .. py:attribute:: AVG
      :canonical: bocoel.core.exams.stats.acc.AccType.AVG
      :value: 'AVERAGE'

      .. autodoc2-docstring:: bocoel.core.exams.stats.acc.AccType.AVG

.. py:class:: Accumulation(typ: bocoel.core.exams.stats.acc.AccType)
   :canonical: bocoel.core.exams.stats.acc.Accumulation

   Bases: :py:obj:`bocoel.core.exams.interfaces.Exam`

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc.Accumulation

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc.Accumulation.__init__

   .. py:method:: _run(index: bocoel.corpora.Index, results: collections.OrderedDict[int, float]) -> numpy.typing.NDArray
      :canonical: bocoel.core.exams.stats.acc.Accumulation._run

   .. py:method:: _acc(array: numpy.typing.NDArray, accumulate: collections.abc.Callable[[numpy.typing.NDArray], numpy.typing.NDArray]) -> numpy.typing.NDArray
      :canonical: bocoel.core.exams.stats.acc.Accumulation._acc
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.exams.stats.acc.Accumulation._acc

.. py:function:: _check_dim(array: numpy.typing.NDArray, /, ndim: int) -> None
   :canonical: bocoel.core.exams.stats.acc._check_dim

   .. autodoc2-docstring:: bocoel.core.exams.stats.acc._check_dim

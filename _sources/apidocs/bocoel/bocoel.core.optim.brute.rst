:py:mod:`bocoel.core.optim.brute`
=================================

.. py:module:: bocoel.core.optim.brute

.. autodoc2-docstring:: bocoel.core.optim.brute
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`BruteForceOptimizer <bocoel.core.optim.brute.BruteForceOptimizer>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.optim.brute.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.optim.brute.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.optim.brute.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.optim.brute.LOGGER

.. py:class:: BruteForceOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, total: int, batch_size: int)
   :canonical: bocoel.core.optim.brute.BruteForceOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. py:property:: task
      :canonical: bocoel.core.optim.brute.BruteForceOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:property:: terminate
      :canonical: bocoel.core.optim.brute.BruteForceOptimizer.terminate
      :type: bool

      .. autodoc2-docstring:: bocoel.core.optim.brute.BruteForceOptimizer.terminate

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.brute.BruteForceOptimizer.step

:py:mod:`bocoel.core.optim.random`
==================================

.. py:module:: bocoel.core.optim.random

.. autodoc2-docstring:: bocoel.core.optim.random
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`RandomOptimizer <bocoel.core.optim.random.RandomOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.random.RandomOptimizer
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.optim.random.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.optim.random.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.optim.random.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.optim.random.LOGGER

.. py:class:: RandomOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, samples: int, batch_size: int)
   :canonical: bocoel.core.optim.random.RandomOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. autodoc2-docstring:: bocoel.core.optim.random.RandomOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.random.RandomOptimizer.__init__

   .. py:property:: task
      :canonical: bocoel.core.optim.random.RandomOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:property:: terminate
      :canonical: bocoel.core.optim.random.RandomOptimizer.terminate
      :type: bool

      .. autodoc2-docstring:: bocoel.core.optim.random.RandomOptimizer.terminate

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.random.RandomOptimizer.step

   .. py:method:: _gen_random(samples: int, /) -> list[int]
      :canonical: bocoel.core.optim.random.RandomOptimizer._gen_random

      .. autodoc2-docstring:: bocoel.core.optim.random.RandomOptimizer._gen_random

:py:mod:`bocoel.core.optim.cma.optim`
=====================================

.. py:module:: bocoel.core.optim.cma.optim

.. autodoc2-docstring:: bocoel.core.optim.cma.optim
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PyCMAOptimizer <bocoel.core.optim.cma.optim.PyCMAOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.cma.optim.PyCMAOptimizer
          :summary:

API
~~~

.. py:class:: PyCMAOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, dims: int, samples: int, minimize: bool = True)
   :canonical: bocoel.core.optim.cma.optim.PyCMAOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. autodoc2-docstring:: bocoel.core.optim.cma.optim.PyCMAOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.cma.optim.PyCMAOptimizer.__init__

   .. py:property:: task
      :canonical: bocoel.core.optim.cma.optim.PyCMAOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.cma.optim.PyCMAOptimizer.step

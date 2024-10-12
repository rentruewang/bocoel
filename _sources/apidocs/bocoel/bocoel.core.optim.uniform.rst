:py:mod:`bocoel.core.optim.uniform`
===================================

.. py:module:: bocoel.core.optim.uniform

.. autodoc2-docstring:: bocoel.core.optim.uniform
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`UniformOptimizer <bocoel.core.optim.uniform.UniformOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.uniform.UniformOptimizer
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.optim.uniform.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.optim.uniform.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.optim.uniform.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.optim.uniform.LOGGER

.. py:class:: UniformOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, grids: collections.abc.Sequence[int], batch_size: int)
   :canonical: bocoel.core.optim.uniform.UniformOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. autodoc2-docstring:: bocoel.core.optim.uniform.UniformOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.uniform.UniformOptimizer.__init__

   .. py:property:: task
      :canonical: bocoel.core.optim.uniform.UniformOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.uniform.UniformOptimizer.step

   .. py:method:: _gen_locs(grids: collections.abc.Sequence[int]) -> collections.abc.Generator[numpy.typing.NDArray, None, None]
      :canonical: bocoel.core.optim.uniform.UniformOptimizer._gen_locs

      .. autodoc2-docstring:: bocoel.core.optim.uniform.UniformOptimizer._gen_locs

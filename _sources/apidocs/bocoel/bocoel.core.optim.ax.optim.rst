:py:mod:`bocoel.core.optim.ax.optim`
====================================

.. py:module:: bocoel.core.optim.ax.optim

.. autodoc2-docstring:: bocoel.core.optim.ax.optim
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AxServiceOptimizer <bocoel.core.optim.ax.optim.AxServiceOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`silence_ax <bocoel.core.optim.ax.optim.silence_ax>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.optim.silence_ax
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_KEY <bocoel.core.optim.ax.optim._KEY>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.optim._KEY
          :summary:
   * - :py:obj:`Device <bocoel.core.optim.ax.optim.Device>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.optim.Device
          :summary:

API
~~~

.. py:data:: _KEY
   :canonical: bocoel.core.optim.ax.optim._KEY
   :value: 'EVAL'

   .. autodoc2-docstring:: bocoel.core.optim.ax.optim._KEY

.. py:data:: Device
   :canonical: bocoel.core.optim.ax.optim.Device
   :value: None

   .. autodoc2-docstring:: bocoel.core.optim.ax.optim.Device

.. py:function:: silence_ax()
   :canonical: bocoel.core.optim.ax.optim.silence_ax

   .. autodoc2-docstring:: bocoel.core.optim.ax.optim.silence_ax

.. py:class:: AxServiceOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, sobol_steps: int = 0, device: bocoel.core.optim.ax.optim.Device = 'cpu', workers: int = 1, task: bocoel.core.tasks.Task = Task.EXPLORE, acqf: str | bocoel.core.optim.ax.acquisition.AcquisitionFunc = AcquisitionFunc.AUTO, surrogate: str | bocoel.core.optim.ax.surrogates.SurrogateModel = SurrogateModel.AUTO, surrogate_kwargs: bocoel.core.optim.ax.surrogates.SurrogateOptions | None = None)
   :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer.__repr__

   .. py:property:: task
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer.step

   .. py:method:: _create_experiment(boundary: bocoel.corpora.Boundary) -> None
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer._create_experiment

      .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer._create_experiment

   .. py:method:: _eval_one_query(tidx: int, parameters: dict[str, float]) -> float
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer._eval_one_query

      .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer._eval_one_query

   .. py:method:: _gen_strat(sobol_steps: int) -> ax.modelbridge.generation_strategy.GenerationStrategy
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer._gen_strat

      .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer._gen_strat

   .. py:method:: _terminate_step(steps: list[ax.modelbridge.generation_strategy.GenerationStep]) -> int
      :canonical: bocoel.core.optim.ax.optim.AxServiceOptimizer._terminate_step
      :staticmethod:

      .. autodoc2-docstring:: bocoel.core.optim.ax.optim.AxServiceOptimizer._terminate_step

:py:mod:`bocoel.core.optim.interfaces.optim`
============================================

.. py:module:: bocoel.core.optim.interfaces.optim

.. autodoc2-docstring:: bocoel.core.optim.interfaces.optim
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Optimizer <bocoel.core.optim.interfaces.optim.Optimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.optim.Optimizer
          :summary:

API
~~~

.. py:class:: Optimizer(index_eval: bocoel.core.optim.interfaces.evals.IndexEvaluator, index: bocoel.corpora.Index, **kwargs: typing.Any)
   :canonical: bocoel.core.optim.interfaces.optim.Optimizer

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.optim.Optimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.optim.Optimizer.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.core.optim.interfaces.optim.Optimizer.__repr__

   .. py:property:: task
      :canonical: bocoel.core.optim.interfaces.optim.Optimizer.task
      :abstractmethod:
      :type: bocoel.core.tasks.Task

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.optim.Optimizer.task

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.interfaces.optim.Optimizer.step
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.optim.Optimizer.step

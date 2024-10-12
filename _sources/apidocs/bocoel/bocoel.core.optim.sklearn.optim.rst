:py:mod:`bocoel.core.optim.sklearn.optim`
=========================================

.. py:module:: bocoel.core.optim.sklearn.optim

.. autodoc2-docstring:: bocoel.core.optim.sklearn.optim
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ScikitLearnCluster <bocoel.core.optim.sklearn.optim.ScikitLearnCluster>`
     - .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnCluster
          :summary:
   * - :py:obj:`ScikitLearnOptimizer <bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer
          :summary:

API
~~~

.. py:class:: ScikitLearnCluster
   :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnCluster

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnCluster

   .. py:attribute:: cluster_centers_
      :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.cluster_centers_
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.cluster_centers_

   .. py:method:: fit(X: typing.Any) -> None
      :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.fit
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.fit

   .. py:method:: predict(X: typing.Any) -> list[int] | numpy.typing.NDArray
      :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.predict
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnCluster.predict

.. py:class:: ScikitLearnOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, embeddings: numpy.typing.NDArray, model: bocoel.core.optim.sklearn.optim.ScikitLearnCluster, batch_size: int)
   :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer

   Bases: :py:obj:`bocoel.core.optim.interfaces.Optimizer`

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer.__init__

   .. py:property:: task
      :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer.task
      :type: bocoel.core.tasks.Task

   .. py:method:: step() -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer.step

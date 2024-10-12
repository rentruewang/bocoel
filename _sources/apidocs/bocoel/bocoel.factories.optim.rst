:py:mod:`bocoel.factories.optim`
================================

.. py:module:: bocoel.factories.optim

.. autodoc2-docstring:: bocoel.factories.optim
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`OptimizerName <bocoel.factories.optim.OptimizerName>`
     - .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`optimizer <bocoel.factories.optim.optimizer>`
     - .. autodoc2-docstring:: bocoel.factories.optim.optimizer
          :summary:

API
~~~

.. py:class:: OptimizerName()
   :canonical: bocoel.factories.optim.OptimizerName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.__init__

   .. py:attribute:: BAYESIAN
      :canonical: bocoel.factories.optim.OptimizerName.BAYESIAN
      :value: 'BAYESIAN'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.BAYESIAN

   .. py:attribute:: KMEANS
      :canonical: bocoel.factories.optim.OptimizerName.KMEANS
      :value: 'KMEANS'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.KMEANS

   .. py:attribute:: KMEDOIDS
      :canonical: bocoel.factories.optim.OptimizerName.KMEDOIDS
      :value: 'KMEDOIDS'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.KMEDOIDS

   .. py:attribute:: RANDOM
      :canonical: bocoel.factories.optim.OptimizerName.RANDOM
      :value: 'RANDOM'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.RANDOM

   .. py:attribute:: BRUTE
      :canonical: bocoel.factories.optim.OptimizerName.BRUTE
      :value: 'BRUTE'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.BRUTE

   .. py:attribute:: UNIFORM
      :canonical: bocoel.factories.optim.OptimizerName.UNIFORM
      :value: 'UNIFORM'

      .. autodoc2-docstring:: bocoel.factories.optim.OptimizerName.UNIFORM

.. py:function:: optimizer(name: str | bocoel.factories.optim.OptimizerName, /, *, corpus: bocoel.Corpus, adaptor: bocoel.Adaptor, **kwargs: typing.Any) -> bocoel.Optimizer
   :canonical: bocoel.factories.optim.optimizer

   .. autodoc2-docstring:: bocoel.factories.optim.optimizer

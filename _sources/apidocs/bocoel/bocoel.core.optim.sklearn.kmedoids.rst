:py:mod:`bocoel.core.optim.sklearn.kmedoids`
============================================

.. py:module:: bocoel.core.optim.sklearn.kmedoids

.. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`KMedoidsOptions <bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions>`
     -
   * - :py:obj:`KMedoidsOptimizer <bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer
          :summary:

API
~~~

.. py:class:: KMedoidsOptions()
   :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions

   Bases: :py:obj:`typing.TypedDict`

   .. py:attribute:: n_clusters
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.n_clusters
      :type: int
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.n_clusters

   .. py:attribute:: metrics
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.metrics
      :type: typing_extensions.NotRequired[str]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.metrics

   .. py:attribute:: method
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.method
      :type: typing_extensions.NotRequired[typing.Literal[alternate, pam]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.method

   .. py:attribute:: init
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.init
      :type: typing_extensions.NotRequired[typing.Literal[random, heuristic, kmedoids++, build]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.init

   .. py:attribute:: max_iter
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.max_iter
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.max_iter

   .. py:attribute:: random_state
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.random_state
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions.random_state

.. py:class:: KMedoidsOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, batch_size: int, embeddings: numpy.typing.NDArray, model_kwargs: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptions)
   :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer

   Bases: :py:obj:`bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer`

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.core.optim.sklearn.kmedoids.KMedoidsOptimizer.__repr__

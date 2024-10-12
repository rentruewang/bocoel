:py:mod:`bocoel.core.optim.sklearn.kmeans`
==========================================

.. py:module:: bocoel.core.optim.sklearn.kmeans

.. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`KMeansOptions <bocoel.core.optim.sklearn.kmeans.KMeansOptions>`
     -
   * - :py:obj:`KMeansOptimizer <bocoel.core.optim.sklearn.kmeans.KMeansOptimizer>`
     - .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptimizer
          :summary:

API
~~~

.. py:class:: KMeansOptions()
   :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions

   Bases: :py:obj:`typing.TypedDict`

   .. py:attribute:: n_clusters
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.n_clusters
      :type: int
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.n_clusters

   .. py:attribute:: init
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.init
      :type: typing_extensions.NotRequired[typing.Literal[k-means++, random]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.init

   .. py:attribute:: n_init
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.n_init
      :type: typing_extensions.NotRequired[int | typing.Literal[auto]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.n_init

   .. py:attribute:: tol
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.tol
      :type: typing_extensions.NotRequired[float]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.tol

   .. py:attribute:: verbose
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.verbose
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.verbose

   .. py:attribute:: random_state
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.random_state
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.random_state

   .. py:attribute:: algorithm
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptions.algorithm
      :type: typing_extensions.NotRequired[typing.Literal[llyod, elkan]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptions.algorithm

.. py:class:: KMeansOptimizer(index_eval: bocoel.core.optim.interfaces.IndexEvaluator, index: bocoel.corpora.Index, *, batch_size: int, embeddings: numpy.typing.NDArray, model_kwargs: bocoel.core.optim.sklearn.kmeans.KMeansOptions)
   :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptimizer

   Bases: :py:obj:`bocoel.core.optim.sklearn.optim.ScikitLearnOptimizer`

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptimizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.sklearn.kmeans.KMeansOptimizer.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.core.optim.sklearn.kmeans.KMeansOptimizer.__repr__

:py:mod:`bocoel.visual.reducers.pca`
====================================

.. py:module:: bocoel.visual.reducers.pca

.. autodoc2-docstring:: bocoel.visual.reducers.pca
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`PCAReducer <bocoel.visual.reducers.pca.PCAReducer>`
     -

API
~~~

.. py:class:: PCAReducer(scores: numpy.typing.ArrayLike = random.rand(100), size: float = 0.5, sample_size: numpy.typing.ArrayLike = np.arange(1, 101).tolist(), desc: collections.abc.Sequence[str] = (), algo: str = 'PCA')
   :canonical: bocoel.visual.reducers.pca.PCAReducer

   Bases: :py:obj:`bocoel.visual.reducers.interfaces.Reducer`

   .. py:method:: reduce_2d(X: numpy.typing.NDArray) -> numpy.typing.NDArray
      :canonical: bocoel.visual.reducers.pca.PCAReducer.reduce_2d

      .. autodoc2-docstring:: bocoel.visual.reducers.pca.PCAReducer.reduce_2d

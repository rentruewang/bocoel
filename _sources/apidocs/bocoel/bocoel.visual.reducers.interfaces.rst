:py:mod:`bocoel.visual.reducers.interfaces`
===========================================

.. py:module:: bocoel.visual.reducers.interfaces

.. autodoc2-docstring:: bocoel.visual.reducers.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Reducer <bocoel.visual.reducers.interfaces.Reducer>`
     -

API
~~~

.. py:class:: Reducer
   :canonical: bocoel.visual.reducers.interfaces.Reducer

   Bases: :py:obj:`typing.Protocol`

   .. py:attribute:: size
      :canonical: bocoel.visual.reducers.interfaces.Reducer.size
      :type: float
      :value: None

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.size

   .. py:attribute:: scores
      :canonical: bocoel.visual.reducers.interfaces.Reducer.scores
      :type: numpy.typing.NDArray
      :value: None

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.scores

   .. py:attribute:: sample_size
      :canonical: bocoel.visual.reducers.interfaces.Reducer.sample_size
      :type: collections.abc.Sequence[int]
      :value: None

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.sample_size

   .. py:attribute:: description
      :canonical: bocoel.visual.reducers.interfaces.Reducer.description
      :type: collections.abc.Sequence[str]
      :value: None

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.description

   .. py:method:: reduce_2d(X: numpy.typing.NDArray) -> numpy.typing.NDArray
      :canonical: bocoel.visual.reducers.interfaces.Reducer.reduce_2d
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.reduce_2d

   .. py:method:: process(X: numpy.typing.NDArray) -> pandas.DataFrame
      :canonical: bocoel.visual.reducers.interfaces.Reducer.process

      .. autodoc2-docstring:: bocoel.visual.reducers.interfaces.Reducer.process

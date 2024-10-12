:py:mod:`bocoel.core.optim.interfaces.evals`
============================================

.. py:module:: bocoel.core.optim.interfaces.evals

.. autodoc2-docstring:: bocoel.core.optim.interfaces.evals
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`SearchEvaluator <bocoel.core.optim.interfaces.evals.SearchEvaluator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.SearchEvaluator
          :summary:
   * - :py:obj:`QueryEvaluator <bocoel.core.optim.interfaces.evals.QueryEvaluator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.QueryEvaluator
          :summary:
   * - :py:obj:`IndexEvaluator <bocoel.core.optim.interfaces.evals.IndexEvaluator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.IndexEvaluator
          :summary:
   * - :py:obj:`CachedIndexEvaluator <bocoel.core.optim.interfaces.evals.CachedIndexEvaluator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.CachedIndexEvaluator
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.core.optim.interfaces.evals.LOGGER>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.core.optim.interfaces.evals.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.LOGGER

.. py:class:: SearchEvaluator
   :canonical: bocoel.core.optim.interfaces.evals.SearchEvaluator

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.SearchEvaluator

   .. py:method:: __call__(results: collections.abc.Mapping[int, bocoel.corpora.SearchResult], /) -> collections.abc.Mapping[int, float]
      :canonical: bocoel.core.optim.interfaces.evals.SearchEvaluator.__call__
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.SearchEvaluator.__call__

.. py:class:: QueryEvaluator
   :canonical: bocoel.core.optim.interfaces.evals.QueryEvaluator

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.QueryEvaluator

   .. py:method:: __call__(query: numpy.typing.ArrayLike, /) -> collections.OrderedDict[int, float]
      :canonical: bocoel.core.optim.interfaces.evals.QueryEvaluator.__call__
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.QueryEvaluator.__call__

.. py:class:: IndexEvaluator
   :canonical: bocoel.core.optim.interfaces.evals.IndexEvaluator

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.IndexEvaluator

   .. py:method:: __call__(idx: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray
      :canonical: bocoel.core.optim.interfaces.evals.IndexEvaluator.__call__
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.IndexEvaluator.__call__

.. py:class:: CachedIndexEvaluator(index_eval: bocoel.core.optim.interfaces.evals.IndexEvaluator, /)
   :canonical: bocoel.core.optim.interfaces.evals.CachedIndexEvaluator

   Bases: :py:obj:`bocoel.core.optim.interfaces.evals.IndexEvaluator`

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.CachedIndexEvaluator

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.evals.CachedIndexEvaluator.__init__

   .. py:method:: __call__(idx: numpy.typing.ArrayLike, /) -> numpy.typing.NDArray
      :canonical: bocoel.core.optim.interfaces.evals.CachedIndexEvaluator.__call__

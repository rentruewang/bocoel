:py:mod:`bocoel.core.optim.interfaces.utils`
============================================

.. py:module:: bocoel.core.optim.interfaces.utils

.. autodoc2-docstring:: bocoel.core.optim.interfaces.utils
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`_BatchedGeneratorIterator <bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator
          :summary:
   * - :py:obj:`BatchedGenerator <bocoel.core.optim.interfaces.utils.BatchedGenerator>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.BatchedGenerator
          :summary:
   * - :py:obj:`RemainingSteps <bocoel.core.optim.interfaces.utils.RemainingSteps>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`T <bocoel.core.optim.interfaces.utils.T>`
     - .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.T
          :summary:

API
~~~

.. py:data:: T
   :canonical: bocoel.core.optim.interfaces.utils.T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.T

.. py:class:: _BatchedGeneratorIterator(iterable: collections.abc.Iterable[bocoel.core.optim.interfaces.utils.T], batch_size: int, /)
   :canonical: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator

   Bases: :py:obj:`collections.abc.Iterator`\ [\ :py:obj:`list`\ [\ :py:obj:`bocoel.core.optim.interfaces.utils.T`\ ]\ ]

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator.__init__

   .. py:method:: __iter__() -> collections.abc.Iterator[list[bocoel.core.optim.interfaces.utils.T]]
      :canonical: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator.__iter__

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator.__iter__

   .. py:method:: __next__() -> list[bocoel.core.optim.interfaces.utils.T]
      :canonical: bocoel.core.optim.interfaces.utils._BatchedGeneratorIterator.__next__

.. py:class:: BatchedGenerator(iterable: collections.abc.Iterable[bocoel.core.optim.interfaces.utils.T], batch_size: int)
   :canonical: bocoel.core.optim.interfaces.utils.BatchedGenerator

   Bases: :py:obj:`typing.Generic`\ [\ :py:obj:`bocoel.core.optim.interfaces.utils.T`\ ]

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.BatchedGenerator

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.BatchedGenerator.__init__

   .. py:method:: __iter__() -> collections.abc.Iterator[list[bocoel.core.optim.interfaces.utils.T]]
      :canonical: bocoel.core.optim.interfaces.utils.BatchedGenerator.__iter__

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.BatchedGenerator.__iter__

.. py:class:: RemainingSteps(count: int | float)
   :canonical: bocoel.core.optim.interfaces.utils.RemainingSteps

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps.__init__

   .. py:property:: count
      :canonical: bocoel.core.optim.interfaces.utils.RemainingSteps.count
      :type: int | float

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps.count

   .. py:method:: step(size: int = 1) -> None
      :canonical: bocoel.core.optim.interfaces.utils.RemainingSteps.step

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps.step

   .. py:property:: done
      :canonical: bocoel.core.optim.interfaces.utils.RemainingSteps.done
      :type: bool

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps.done

   .. py:method:: infinite() -> typing_extensions.Self
      :canonical: bocoel.core.optim.interfaces.utils.RemainingSteps.infinite
      :classmethod:

      .. autodoc2-docstring:: bocoel.core.optim.interfaces.utils.RemainingSteps.infinite

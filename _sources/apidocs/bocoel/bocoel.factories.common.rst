:py:mod:`bocoel.factories.common`
=================================

.. py:module:: bocoel.factories.common

.. autodoc2-docstring:: bocoel.factories.common
   :allowtitles:

Module Contents
---------------

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`correct_kwargs <bocoel.factories.common.correct_kwargs>`
     - .. autodoc2-docstring:: bocoel.factories.common.correct_kwargs
          :summary:
   * - :py:obj:`auto_device <bocoel.factories.common.auto_device>`
     - .. autodoc2-docstring:: bocoel.factories.common.auto_device
          :summary:
   * - :py:obj:`auto_device_list <bocoel.factories.common.auto_device_list>`
     - .. autodoc2-docstring:: bocoel.factories.common.auto_device_list
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`P <bocoel.factories.common.P>`
     - .. autodoc2-docstring:: bocoel.factories.common.P
          :summary:
   * - :py:obj:`T <bocoel.factories.common.T>`
     - .. autodoc2-docstring:: bocoel.factories.common.T
          :summary:

API
~~~

.. py:data:: P
   :canonical: bocoel.factories.common.P
   :value: 'ParamSpec(...)'

   .. autodoc2-docstring:: bocoel.factories.common.P

.. py:data:: T
   :canonical: bocoel.factories.common.T
   :value: 'TypeVar(...)'

   .. autodoc2-docstring:: bocoel.factories.common.T

.. py:function:: correct_kwargs(function: collections.abc.Callable[bocoel.factories.common.P, bocoel.factories.common.T]) -> collections.abc.Callable[bocoel.factories.common.P, bocoel.factories.common.T]
   :canonical: bocoel.factories.common.correct_kwargs

   .. autodoc2-docstring:: bocoel.factories.common.correct_kwargs

.. py:function:: auto_device(device: str, /) -> str
   :canonical: bocoel.factories.common.auto_device

   .. autodoc2-docstring:: bocoel.factories.common.auto_device

.. py:function:: auto_device_list(device: str, num_models: int, /) -> list[str]
   :canonical: bocoel.factories.common.auto_device_list

   .. autodoc2-docstring:: bocoel.factories.common.auto_device_list

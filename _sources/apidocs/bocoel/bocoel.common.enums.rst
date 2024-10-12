:py:mod:`bocoel.common.enums`
=============================

.. py:module:: bocoel.common.enums

.. autodoc2-docstring:: bocoel.common.enums
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`StrEnum <bocoel.common.enums.StrEnum>`
     -

API
~~~

.. py:exception:: ItemNotFound()
   :canonical: bocoel.common.enums.ItemNotFound

   Bases: :py:obj:`Exception`

.. py:class:: StrEnum()
   :canonical: bocoel.common.enums.StrEnum

   Bases: :py:obj:`str`, :py:obj:`enum.Enum`

   .. py:method:: lookup(name: str | typing_extensions.Self) -> typing_extensions.Self
      :canonical: bocoel.common.enums.StrEnum.lookup
      :classmethod:

      .. autodoc2-docstring:: bocoel.common.enums.StrEnum.lookup

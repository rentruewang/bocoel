:py:mod:`bocoel.models.adaptors.dicts`
======================================

.. py:module:: bocoel.models.adaptors.dicts

.. autodoc2-docstring:: bocoel.models.adaptors.dicts
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AdaptorMapping <bocoel.models.adaptors.dicts.AdaptorMapping>`
     -

API
~~~

.. py:class:: AdaptorMapping(adaptors: collections.abc.Mapping[str, bocoel.models.adaptors.interfaces.Adaptor])
   :canonical: bocoel.models.adaptors.dicts.AdaptorMapping

   Bases: :py:obj:`bocoel.models.adaptors.interfaces.AdaptorBundle`

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Mapping[str, collections.abc.Sequence[float] | numpy.typing.NDArray]
      :canonical: bocoel.models.adaptors.dicts.AdaptorMapping.evaluate

      .. autodoc2-docstring:: bocoel.models.adaptors.dicts.AdaptorMapping.evaluate

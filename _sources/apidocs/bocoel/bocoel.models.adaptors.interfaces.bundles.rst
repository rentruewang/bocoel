:py:mod:`bocoel.models.adaptors.interfaces.bundles`
===================================================

.. py:module:: bocoel.models.adaptors.interfaces.bundles

.. autodoc2-docstring:: bocoel.models.adaptors.interfaces.bundles
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AdaptorBundle <bocoel.models.adaptors.interfaces.bundles.AdaptorBundle>`
     -

API
~~~

.. py:class:: AdaptorBundle
   :canonical: bocoel.models.adaptors.interfaces.bundles.AdaptorBundle

   Bases: :py:obj:`typing.Protocol`

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Mapping[str, collections.abc.Sequence[float] | numpy.typing.NDArray]
      :canonical: bocoel.models.adaptors.interfaces.bundles.AdaptorBundle.evaluate
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.models.adaptors.interfaces.bundles.AdaptorBundle.evaluate

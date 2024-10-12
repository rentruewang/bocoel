:py:mod:`bocoel.models.lms.interfaces.generative`
=================================================

.. py:module:: bocoel.models.lms.interfaces.generative

.. autodoc2-docstring:: bocoel.models.lms.interfaces.generative
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`GenerativeModel <bocoel.models.lms.interfaces.generative.GenerativeModel>`
     -

API
~~~

.. py:class:: GenerativeModel
   :canonical: bocoel.models.lms.interfaces.generative.GenerativeModel

   Bases: :py:obj:`typing.Protocol`

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.lms.interfaces.generative.GenerativeModel.__repr__

   .. py:method:: generate(prompts: collections.abc.Sequence[str], /) -> collections.abc.Sequence[str]
      :canonical: bocoel.models.lms.interfaces.generative.GenerativeModel.generate
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.models.lms.interfaces.generative.GenerativeModel.generate

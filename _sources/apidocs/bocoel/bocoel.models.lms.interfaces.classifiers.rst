:py:mod:`bocoel.models.lms.interfaces.classifiers`
==================================================

.. py:module:: bocoel.models.lms.interfaces.classifiers

.. autodoc2-docstring:: bocoel.models.lms.interfaces.classifiers
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`ClassifierModel <bocoel.models.lms.interfaces.classifiers.ClassifierModel>`
     -

API
~~~

.. py:class:: ClassifierModel
   :canonical: bocoel.models.lms.interfaces.classifiers.ClassifierModel

   Bases: :py:obj:`typing.Protocol`

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.lms.interfaces.classifiers.ClassifierModel.__repr__

   .. py:method:: classify(prompts: collections.abc.Sequence[str], /) -> numpy.typing.NDArray
      :canonical: bocoel.models.lms.interfaces.classifiers.ClassifierModel.classify

      .. autodoc2-docstring:: bocoel.models.lms.interfaces.classifiers.ClassifierModel.classify

   .. py:method:: _classify(prompts: collections.abc.Sequence[str], /) -> numpy.typing.NDArray
      :canonical: bocoel.models.lms.interfaces.classifiers.ClassifierModel._classify
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.models.lms.interfaces.classifiers.ClassifierModel._classify

   .. py:property:: choices
      :canonical: bocoel.models.lms.interfaces.classifiers.ClassifierModel.choices
      :abstractmethod:
      :type: collections.abc.Sequence[str]

      .. autodoc2-docstring:: bocoel.models.lms.interfaces.classifiers.ClassifierModel.choices

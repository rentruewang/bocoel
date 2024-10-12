:py:mod:`bocoel.core.exams.examinators`
=======================================

.. py:module:: bocoel.core.exams.examinators

.. autodoc2-docstring:: bocoel.core.exams.examinators
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Examinator <bocoel.core.exams.examinators.Examinator>`
     - .. autodoc2-docstring:: bocoel.core.exams.examinators.Examinator
          :summary:

API
~~~

.. py:class:: Examinator(exams: collections.abc.Mapping[str, bocoel.core.exams.interfaces.Exam])
   :canonical: bocoel.core.exams.examinators.Examinator

   .. autodoc2-docstring:: bocoel.core.exams.examinators.Examinator

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.core.exams.examinators.Examinator.__init__

   .. py:method:: examine(index: bocoel.corpora.Index, results: collections.OrderedDict[int, float]) -> pandas.DataFrame
      :canonical: bocoel.core.exams.examinators.Examinator.examine

      .. autodoc2-docstring:: bocoel.core.exams.examinators.Examinator.examine

   .. py:method:: presets() -> typing_extensions.Self
      :canonical: bocoel.core.exams.examinators.Examinator.presets
      :classmethod:

      .. autodoc2-docstring:: bocoel.core.exams.examinators.Examinator.presets

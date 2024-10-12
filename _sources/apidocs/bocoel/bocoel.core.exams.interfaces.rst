:py:mod:`bocoel.core.exams.interfaces`
======================================

.. py:module:: bocoel.core.exams.interfaces

.. autodoc2-docstring:: bocoel.core.exams.interfaces
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Exam <bocoel.core.exams.interfaces.Exam>`
     - .. autodoc2-docstring:: bocoel.core.exams.interfaces.Exam
          :summary:

API
~~~

.. py:class:: Exam
   :canonical: bocoel.core.exams.interfaces.Exam

   Bases: :py:obj:`typing.Protocol`

   .. autodoc2-docstring:: bocoel.core.exams.interfaces.Exam

   .. py:method:: run(index: bocoel.corpora.Index, results: collections.OrderedDict[int, float]) -> numpy.typing.NDArray
      :canonical: bocoel.core.exams.interfaces.Exam.run

      .. autodoc2-docstring:: bocoel.core.exams.interfaces.Exam.run

   .. py:method:: _run(index: bocoel.corpora.Index, results: collections.OrderedDict[int, float]) -> numpy.typing.NDArray
      :canonical: bocoel.core.exams.interfaces.Exam._run
      :abstractmethod:

      .. autodoc2-docstring:: bocoel.core.exams.interfaces.Exam._run

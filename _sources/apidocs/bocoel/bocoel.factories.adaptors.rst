:py:mod:`bocoel.factories.adaptors`
===================================

.. py:module:: bocoel.factories.adaptors

.. autodoc2-docstring:: bocoel.factories.adaptors
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`AdaptorName <bocoel.factories.adaptors.AdaptorName>`
     - .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`adaptor <bocoel.factories.adaptors.adaptor>`
     - .. autodoc2-docstring:: bocoel.factories.adaptors.adaptor
          :summary:

API
~~~

.. py:class:: AdaptorName()
   :canonical: bocoel.factories.adaptors.AdaptorName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName.__init__

   .. py:attribute:: BIGBENCH_MC
      :canonical: bocoel.factories.adaptors.AdaptorName.BIGBENCH_MC
      :value: 'BIGBENCH_MULTIPLE_CHOICE'

      .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName.BIGBENCH_MC

   .. py:attribute:: BIGBENCH_QA
      :canonical: bocoel.factories.adaptors.AdaptorName.BIGBENCH_QA
      :value: 'BIGBENCH_QUESTION_ANSWER'

      .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName.BIGBENCH_QA

   .. py:attribute:: SST2
      :canonical: bocoel.factories.adaptors.AdaptorName.SST2
      :value: 'SST2'

      .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName.SST2

   .. py:attribute:: GLUE
      :canonical: bocoel.factories.adaptors.AdaptorName.GLUE
      :value: 'GLUE'

      .. autodoc2-docstring:: bocoel.factories.adaptors.AdaptorName.GLUE

.. py:function:: adaptor(name: str | bocoel.factories.adaptors.AdaptorName, /, **kwargs: typing.Any) -> bocoel.Adaptor
   :canonical: bocoel.factories.adaptors.adaptor

   .. autodoc2-docstring:: bocoel.factories.adaptors.adaptor

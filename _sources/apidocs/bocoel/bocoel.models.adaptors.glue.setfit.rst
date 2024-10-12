:py:mod:`bocoel.models.adaptors.glue.setfit`
============================================

.. py:module:: bocoel.models.adaptors.glue.setfit

.. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`GlueAdaptor <bocoel.models.adaptors.glue.setfit.GlueAdaptor>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.GlueAdaptor
          :summary:

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`LOGGER <bocoel.models.adaptors.glue.setfit.LOGGER>`
     - .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.LOGGER
          :summary:

API
~~~

.. py:data:: LOGGER
   :canonical: bocoel.models.adaptors.glue.setfit.LOGGER
   :value: 'get_logger(...)'

   .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.LOGGER

.. py:class:: GlueAdaptor(lm: bocoel.models.lms.ClassifierModel, texts: str = 'text', label: str = 'label', label_text: str = 'label_text', choices: collections.abc.Sequence[str] = ('negative', 'positive'))
   :canonical: bocoel.models.adaptors.glue.setfit.GlueAdaptor

   Bases: :py:obj:`bocoel.models.adaptors.interfaces.Adaptor`

   .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.GlueAdaptor

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.GlueAdaptor.__init__

   .. py:method:: evaluate(data: collections.abc.Mapping[str, collections.abc.Sequence[typing.Any]]) -> collections.abc.Sequence[float] | numpy.typing.NDArray
      :canonical: bocoel.models.adaptors.glue.setfit.GlueAdaptor.evaluate

   .. py:method:: task_choices(name: typing.Literal[sst2, mrpc, mnli, qqp, rte, qnli], split: typing.Literal[train, validation, test]) -> collections.abc.Sequence[str]
      :canonical: bocoel.models.adaptors.glue.setfit.GlueAdaptor.task_choices
      :staticmethod:

      .. autodoc2-docstring:: bocoel.models.adaptors.glue.setfit.GlueAdaptor.task_choices

:py:mod:`bocoel.factories.lms`
==============================

.. py:module:: bocoel.factories.lms

.. autodoc2-docstring:: bocoel.factories.lms
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`GeneratorName <bocoel.factories.lms.GeneratorName>`
     - .. autodoc2-docstring:: bocoel.factories.lms.GeneratorName
          :summary:
   * - :py:obj:`ClassifierName <bocoel.factories.lms.ClassifierName>`
     - .. autodoc2-docstring:: bocoel.factories.lms.ClassifierName
          :summary:

Functions
~~~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`generative <bocoel.factories.lms.generative>`
     - .. autodoc2-docstring:: bocoel.factories.lms.generative
          :summary:
   * - :py:obj:`classifier <bocoel.factories.lms.classifier>`
     - .. autodoc2-docstring:: bocoel.factories.lms.classifier
          :summary:

API
~~~

.. py:class:: GeneratorName()
   :canonical: bocoel.factories.lms.GeneratorName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.lms.GeneratorName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.lms.GeneratorName.__init__

   .. py:attribute:: HUGGINGFACE_GENERATIVE
      :canonical: bocoel.factories.lms.GeneratorName.HUGGINGFACE_GENERATIVE
      :value: 'HUGGINGFACE_GENERATIVE'

      .. autodoc2-docstring:: bocoel.factories.lms.GeneratorName.HUGGINGFACE_GENERATIVE

.. py:class:: ClassifierName()
   :canonical: bocoel.factories.lms.ClassifierName

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. autodoc2-docstring:: bocoel.factories.lms.ClassifierName

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.factories.lms.ClassifierName.__init__

   .. py:attribute:: HUGGINGFACE_LOGITS
      :canonical: bocoel.factories.lms.ClassifierName.HUGGINGFACE_LOGITS
      :value: 'HUGGINGFACE_LOGITS'

      .. autodoc2-docstring:: bocoel.factories.lms.ClassifierName.HUGGINGFACE_LOGITS

   .. py:attribute:: HUGGINGFACE_SEQUENCE
      :canonical: bocoel.factories.lms.ClassifierName.HUGGINGFACE_SEQUENCE
      :value: 'HUGGINGFACE_SEQUENCE'

      .. autodoc2-docstring:: bocoel.factories.lms.ClassifierName.HUGGINGFACE_SEQUENCE

.. py:function:: generative(name: str | bocoel.factories.lms.GeneratorName, /, *, model_path: str, batch_size: int, device: str = 'auto', add_sep_token: bool = False) -> bocoel.GenerativeModel
   :canonical: bocoel.factories.lms.generative

   .. autodoc2-docstring:: bocoel.factories.lms.generative

.. py:function:: classifier(name: str | bocoel.factories.lms.ClassifierName, /, *, model_path: str, batch_size: int, choices: collections.abc.Sequence[str], device: str = 'auto', add_sep_token: bool = False) -> bocoel.ClassifierModel
   :canonical: bocoel.factories.lms.classifier

   .. autodoc2-docstring:: bocoel.factories.lms.classifier

:py:mod:`bocoel.models.lms.huggingface.sequences`
=================================================

.. py:module:: bocoel.models.lms.huggingface.sequences

.. autodoc2-docstring:: bocoel.models.lms.huggingface.sequences
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceSequenceLM <bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM>`
     - .. autodoc2-docstring:: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM
          :summary:

API
~~~

.. py:class:: HuggingfaceSequenceLM(model_path: str, device: str, choices: collections.abc.Sequence[str], add_sep_token: bool = False)
   :canonical: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM

   Bases: :py:obj:`bocoel.models.lms.interfaces.ClassifierModel`

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM.__repr__

   .. py:property:: choices
      :canonical: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM.choices
      :type: collections.abc.Sequence[str]

   .. py:method:: _classify(prompts: collections.abc.Sequence[str], /) -> numpy.typing.NDArray
      :canonical: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM._classify

   .. py:method:: to(device: str, /) -> typing_extensions.Self
      :canonical: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM.to

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.sequences.HuggingfaceSequenceLM.to

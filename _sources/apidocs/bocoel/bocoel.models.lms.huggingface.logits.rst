:py:mod:`bocoel.models.lms.huggingface.logits`
==============================================

.. py:module:: bocoel.models.lms.huggingface.logits

.. autodoc2-docstring:: bocoel.models.lms.huggingface.logits
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceLogitsLM <bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM>`
     - .. autodoc2-docstring:: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM
          :summary:

API
~~~

.. py:class:: HuggingfaceLogitsLM(model_path: str, batch_size: int, device: str, choices: collections.abc.Sequence[str], add_sep_token: bool = False)
   :canonical: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM

   Bases: :py:obj:`bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM`, :py:obj:`bocoel.models.lms.interfaces.ClassifierModel`

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM.__init__

   .. py:property:: choices
      :canonical: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM.choices
      :type: collections.abc.Sequence[str]

   .. py:method:: _classify(prompts: collections.abc.Sequence[str], /) -> numpy.typing.NDArray
      :canonical: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM._classify

   .. py:method:: _encode_tokens(tokens: collections.abc.Sequence[str]) -> collections.abc.Sequence[int]
      :canonical: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM._encode_tokens

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.logits.HuggingfaceLogitsLM._encode_tokens

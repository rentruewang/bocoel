:py:mod:`bocoel.models.lms.huggingface.generative`
==================================================

.. py:module:: bocoel.models.lms.huggingface.generative

.. autodoc2-docstring:: bocoel.models.lms.huggingface.generative
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceGenerativeLM <bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM>`
     - .. autodoc2-docstring:: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM
          :summary:

API
~~~

.. py:class:: HuggingfaceGenerativeLM(model_path: str, batch_size: int, device: str, add_sep_token: bool = False)
   :canonical: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM

   Bases: :py:obj:`bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM`, :py:obj:`bocoel.models.lms.interfaces.GenerativeModel`

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM.__init__

   .. py:method:: generate(prompts: collections.abc.Sequence[str], /) -> collections.abc.Sequence[str]
      :canonical: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM.generate

   .. py:method:: _generate_batch(prompts: collections.abc.Sequence[str]) -> list[str]
      :canonical: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM._generate_batch

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.generative.HuggingfaceGenerativeLM._generate_batch

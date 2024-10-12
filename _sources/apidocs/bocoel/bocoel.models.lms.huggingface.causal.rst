:py:mod:`bocoel.models.lms.huggingface.causal`
==============================================

.. py:module:: bocoel.models.lms.huggingface.causal

.. autodoc2-docstring:: bocoel.models.lms.huggingface.causal
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceCausalLM <bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM>`
     - .. autodoc2-docstring:: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM
          :summary:

API
~~~

.. py:class:: HuggingfaceCausalLM(model_path: str, batch_size: int, device: str, add_sep_token: bool = False)
   :canonical: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.__init__

   .. py:method:: __repr__() -> str
      :canonical: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.__repr__

   .. py:method:: to(device: str, /) -> typing_extensions.Self
      :canonical: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.to

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.to

   .. py:property:: device
      :canonical: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.device
      :type: str

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.causal.HuggingfaceCausalLM.device

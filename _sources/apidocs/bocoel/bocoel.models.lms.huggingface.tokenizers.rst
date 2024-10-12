:py:mod:`bocoel.models.lms.huggingface.tokenizers`
==================================================

.. py:module:: bocoel.models.lms.huggingface.tokenizers

.. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`HuggingfaceTokenizer <bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer>`
     - .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer
          :summary:

API
~~~

.. py:class:: HuggingfaceTokenizer(model_path: str, device: str, add_sep_token: bool)
   :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer

   .. rubric:: Initialization

   .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.__init__

   .. py:method:: to(device: str, /) -> typing_extensions.Self
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.to

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.to

   .. py:method:: tokenize(prompts: collections.abc.Sequence[str], /, max_length: int | None = None)
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.tokenize

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.tokenize

   .. py:method:: __call__(prompts: collections.abc.Sequence[str], /, max_length: int | None = None)
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.__call__

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.__call__

   .. py:method:: encode(prompts: collections.abc.Sequence[str], /, return_tensors: str | None = None, add_special_tokens: bool = True)
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.encode

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.encode

   .. py:method:: decode(outputs: typing.Any, /, skip_special_tokens: bool = True) -> str
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.decode

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.decode

   .. py:method:: batch_decode(outputs: typing.Any, /, skip_special_tokens: bool = True) -> list[str]
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.batch_decode

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.batch_decode

   .. py:property:: device
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.device
      :type: str

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.device

   .. py:property:: pad_token_id
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.pad_token_id
      :type: int

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.pad_token_id

   .. py:property:: pad_token
      :canonical: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.pad_token
      :type: str

      .. autodoc2-docstring:: bocoel.models.lms.huggingface.tokenizers.HuggingfaceTokenizer.pad_token

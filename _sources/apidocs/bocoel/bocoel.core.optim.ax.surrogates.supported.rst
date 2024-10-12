:py:mod:`bocoel.core.optim.ax.surrogates.supported`
===================================================

.. py:module:: bocoel.core.optim.ax.surrogates.supported

.. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`MLLOptions <bocoel.core.optim.ax.surrogates.supported.MLLOptions>`
     -
   * - :py:obj:`SurrogateOptions <bocoel.core.optim.ax.surrogates.supported.SurrogateOptions>`
     -
   * - :py:obj:`SurrogateModel <bocoel.core.optim.ax.surrogates.supported.SurrogateModel>`
     -

API
~~~

.. py:class:: MLLOptions()
   :canonical: bocoel.core.optim.ax.surrogates.supported.MLLOptions

   Bases: :py:obj:`typing.TypedDict`

   .. py:attribute:: num_samples
      :canonical: bocoel.core.optim.ax.surrogates.supported.MLLOptions.num_samples
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.MLLOptions.num_samples

   .. py:attribute:: warmup_steps
      :canonical: bocoel.core.optim.ax.surrogates.supported.MLLOptions.warmup_steps
      :type: typing_extensions.NotRequired[int]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.MLLOptions.warmup_steps

.. py:class:: SurrogateOptions()
   :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions

   Bases: :py:obj:`typing.TypedDict`

   .. py:attribute:: mll_class
      :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions.mll_class
      :type: typing_extensions.NotRequired[type[gpytorch.mlls.marginal_log_likelihood.MarginalLogLikelihood]]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions.mll_class

   .. py:attribute:: mll_options
      :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions.mll_options
      :type: typing_extensions.NotRequired[bocoel.core.optim.ax.surrogates.supported.MLLOptions]
      :value: None

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions.mll_options

.. py:class:: SurrogateModel()
   :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateModel

   Bases: :py:obj:`bocoel.common.StrEnum`

   .. py:attribute:: SAAS
      :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.SAAS
      :value: 'SAAS'

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.SAAS

   .. py:attribute:: AUTO
      :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.AUTO
      :value: 'AUTO'

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.AUTO

   .. py:method:: surrogate(surrogate_options: bocoel.core.optim.ax.surrogates.supported.SurrogateOptions | None) -> ax.models.torch.botorch_modular.surrogate.Surrogate | None
      :canonical: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.surrogate

      .. autodoc2-docstring:: bocoel.core.optim.ax.surrogates.supported.SurrogateModel.surrogate

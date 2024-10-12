:py:mod:`bocoel.core.optim.ax.acquisition.entropy`
==================================================

.. py:module:: bocoel.core.optim.ax.acquisition.entropy

.. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.entropy
   :allowtitles:

Module Contents
---------------

Classes
~~~~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`Entropy <bocoel.core.optim.ax.acquisition.entropy.Entropy>`
     -

Data
~~~~

.. list-table::
   :class: autosummary longtable
   :align: left

   * - :py:obj:`__ACQF_REGISTER_ENTROPY <bocoel.core.optim.ax.acquisition.entropy.__ACQF_REGISTER_ENTROPY>`
     - .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.entropy.__ACQF_REGISTER_ENTROPY
          :summary:

API
~~~

.. py:class:: Entropy(model: botorch.models.model.Model, candidate_set: torch.Tensor, num_fantasies: int = 16, num_mv_samples: int = 10, num_y_samples: int = 128, posterior_transform: typing.Optional[botorch.acquisition.objective.PosteriorTransform] = None, use_gumbel: bool = True, maximize: bool = True, X_pending: typing.Optional[torch.Tensor] = None, train_inputs: typing.Optional[torch.Tensor] = None)
   :canonical: bocoel.core.optim.ax.acquisition.entropy.Entropy

   Bases: :py:obj:`botorch.acquisition.qMaxValueEntropy`

   .. py:method:: _compute_information_gain(X: torch.Tensor, mean_M: torch.Tensor, variance_M: torch.Tensor, covar_mM: torch.Tensor) -> torch.Tensor
      :canonical: bocoel.core.optim.ax.acquisition.entropy.Entropy._compute_information_gain

.. py:data:: __ACQF_REGISTER_ENTROPY
   :canonical: bocoel.core.optim.ax.acquisition.entropy.__ACQF_REGISTER_ENTROPY
   :value: 'acqf_input_constructor(...)'

   .. autodoc2-docstring:: bocoel.core.optim.ax.acquisition.entropy.__ACQF_REGISTER_ENTROPY

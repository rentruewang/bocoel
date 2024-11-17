# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from botorch.acquisition import input_constructors, qMaxValueEntropy
from torch import Tensor


class Entropy(qMaxValueEntropy):
    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        # Unused variables.
        _ = mean_M
        _ = variance_M
        _ = covar_mM

        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.distribution.entropy()  # batch_shape x num_fantasies x (m)

        if self.posterior_max_values.ndim == 2:
            permute_idcs = [-1, *range(H0.ndim - 1)]
        else:
            permute_idcs = [-2, *range(H0.ndim - 2), -1]
        ig = H0.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        return ig


__ACQF_REGISTER_ENTROPY = input_constructors.acqf_input_constructor(Entropy)
__ACQF_REGISTER_ENTROPY(input_constructors.construct_inputs_qMES)

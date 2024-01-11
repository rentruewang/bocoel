from typing import Any

from botorch.acquisition import qMaxValueEntropy
from botorch.models import utils
from torch import Tensor


class MaxEntropy(qMaxValueEntropy):
    """
    @doctryucsd
    FIXME: This is not currently accomplishing computing the maximum value entropy function.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    def _compute_information_gain(
        self, X: Tensor, mean_M: Tensor, variance_M: Tensor, covar_mM: Tensor
    ) -> Tensor:
        # compute the std_m, variance_m with noisy observation
        posterior_m = self.model.posterior(
            X.unsqueeze(-3),
            observation_noise=True,
            posterior_transform=self.posterior_transform,
        )

        # batch_shape x num_fantasies x (m) x (1 + num_trace_observations)
        variance_m = posterior_m.distribution.covariance_matrix
        utils.check_no_nans(variance_m)

        # H0 = H(ym | x, Dt)
        H0 = posterior_m.distribution.entropy()  # batch_shape x num_fantasies x (m)

        if self.posterior_max_values.ndim == 2:
            permute_idcs = [-1, *range(H0.ndim - 1)]
        else:
            permute_idcs = [-2, *range(H0.ndim - 2), -1]
        ig = H0.permute(*permute_idcs)  # num_fantasies x batch_shape x (m)
        return ig

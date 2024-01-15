from typing import TypedDict

from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from typing_extensions import NotRequired

from bocoel.common import StrEnum


class MLLOptions(TypedDict):
    num_samples: NotRequired[int]
    warmup_steps: NotRequired[int]


class SurrogateOptions(TypedDict):
    mll_class: NotRequired[type[MarginalLogLikelihood]]
    mll_options: NotRequired[MLLOptions]


class SurrogateModel(StrEnum):
    SAAS = "SAAS"
    AUTO = "AUTO"

    def surrogate(self, surrogate_options: SurrogateOptions | None) -> Surrogate | None:
        if surrogate_options is None:
            surrogate_options = {}

        match self:
            case SurrogateModel.AUTO:
                return None
            case SurrogateModel.SAAS:
                return Surrogate(
                    botorch_model_class=SaasFullyBayesianSingleTaskGP,
                    **surrogate_options,
                )

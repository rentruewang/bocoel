from typing import Any, TypedDict

from ax.models.torch.botorch_modular.model import SurrogateSpec
from ax.models.torch.botorch_modular.surrogate import Surrogate
from botorch.models.fully_bayesian import SaasFullyBayesianSingleTaskGP
from gpytorch.mlls.marginal_log_likelihood import MarginalLogLikelihood
from typing_extensions import NotRequired

from bocoel.common import StrEnum


class MLLOptions(TypedDict):
    num_samples: NotRequired[int]
    warmup_steps: NotRequired[int]


class SurrogateModel(StrEnum):
    SAAS = "SAAS"
    AUTO = "AUTO"

    def surrogate(
        self,
        mll_class: type[MarginalLogLikelihood] | None = None,
        mll_options: MLLOptions | None = None,
    ) -> Surrogate | None:
        kwargs: dict[str, Any] = {}
        if mll_class:
            kwargs.update({"mll_class": mll_class})
        if mll_options:
            kwargs.update({"mll_options": mll_options})

        match self:
            case SurrogateModel.AUTO:
                return None
            case SurrogateModel.SAAS:
                return Surrogate(
                    botorch_model_class=SaasFullyBayesianSingleTaskGP, **kwargs
                )

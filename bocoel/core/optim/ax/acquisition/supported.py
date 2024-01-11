from enum import Enum

from botorch.acquisition import (
    AcquisitionFunction,
    qExpectedImprovement,
    qMaxValueEntropy,
    qUpperConfidenceBound,
)

from .entropy import Entropy


class AcquisitionFunc(str, Enum):
    MAX_ENTROPY = "entropy"
    MAX_VALUE_ENTROPY = "mes"
    UPPER_CONFIDENCE_BOUND = "ucb"
    EXPECTED_IMPROVEMENT = "ei"

    @property
    def botorch_acqf_class(self) -> type[AcquisitionFunction]:
        match self:
            case AcquisitionFunc.MAX_ENTROPY:
                return Entropy
            case AcquisitionFunc.MAX_VALUE_ENTROPY:
                return qMaxValueEntropy
            case AcquisitionFunc.UPPER_CONFIDENCE_BOUND:
                return qUpperConfidenceBound
            case AcquisitionFunc.EXPECTED_IMPROVEMENT:
                return qExpectedImprovement

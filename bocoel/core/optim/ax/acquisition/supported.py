from botorch.acquisition import (
    AcquisitionFunction,
    ExpectedImprovement,
    UpperConfidenceBound,
    qExpectedImprovement,
    qMaxValueEntropy,
    qUpperConfidenceBound,
)

from bocoel.common import StrEnum

from .entropy import Entropy


class AcquisitionFunc(StrEnum):
    ENTROPY = "ENTROPY"
    ME = "MAX_VALUE_ENTROPY"
    UCB = "UPPER_CONFIDENCE_BOUND"
    QUCB = "Q_UPPER_CONFIDENCE_BOUND"
    EI = "EXPECTED_IMPROVEMENT"
    QEI = "Q_EXPECTED_IMPROVEMENT"

    @property
    def botorch_acqf_class(self) -> type[AcquisitionFunction]:
        match self:
            case AcquisitionFunc.ENTROPY:
                return Entropy
            case AcquisitionFunc.ME:
                return qMaxValueEntropy
            case AcquisitionFunc.UCB:
                return UpperConfidenceBound
            case AcquisitionFunc.QUCB:
                return qUpperConfidenceBound
            case AcquisitionFunc.EI:
                return ExpectedImprovement
            case AcquisitionFunc.QEI:
                return qExpectedImprovement

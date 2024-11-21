# Copyright (c) 2024 RenChu Wang - All Rights Reserved

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
    MES = "MAX_VALUE_ENTROPY"
    UCB = "UPPER_CONFIDENCE_BOUND"
    QUCB = "QUASI_UPPER_CONFIDENCE_BOUND"
    EI = "EXPECTED_IMPROVEMENT"
    QEI = "QUASI_EXPECTED_IMPROVEMENT"
    AUTO = "AUTO"

    @property
    def botorch_acqf_class(self) -> type[AcquisitionFunction] | None:
        match self:
            case AcquisitionFunc.AUTO:
                return None
            case AcquisitionFunc.ENTROPY:
                return Entropy
            case AcquisitionFunc.MES:
                return qMaxValueEntropy
            case AcquisitionFunc.UCB:
                return UpperConfidenceBound
            case AcquisitionFunc.QUCB:
                return qUpperConfidenceBound
            case AcquisitionFunc.EI:
                return ExpectedImprovement
            case AcquisitionFunc.QEI:
                return qExpectedImprovement

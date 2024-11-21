# Copyright (c) 2024 RenChu Wang - All Rights Reserved

"""
Optimizers is much like optimizers in PyTorch,
but for the purpose of optimizing queries and search.
Each optimizer would perform a few steps that collectively
would guide the search towards the optimal trajectory.
"""

from .ax import AcquisitionFunc, AxServiceOptimizer, AxServiceParameter
from .brute import BruteForceOptimizer
from .corpora import CorpusEvaluator
from .interfaces import (
    CachedIndexEvaluator,
    IndexEvaluator,
    Optimizer,
    QueryEvaluator,
    SearchEvaluator,
)
from .interfaces.utils import RemainingSteps
from .random import RandomOptimizer
from .sklearn import (
    KMeansOptimizer,
    KMeansOptions,
    KMedoidsOptimizer,
    KMedoidsOptions,
    ScikitLearnOptimizer,
)
from .uniform import UniformOptimizer

__all__ = [
    "AcquisitionFunc",
    "AxServiceOptimizer",
    "AxServiceParameter",
    "BruteForceOptimizer",
    "CorpusEvaluator",
    "CachedIndexEvaluator",
    "IndexEvaluator",
    "Optimizer",
    "QueryEvaluator",
    "SearchEvaluator",
    "RemainingSteps",
    "RandomOptimizer",
    "KMeansOptimizer",
    "KMeansOptions",
    "KMedoidsOptimizer",
    "KMedoidsOptions",
    "ScikitLearnOptimizer",
    "UniformOptimizer",
]

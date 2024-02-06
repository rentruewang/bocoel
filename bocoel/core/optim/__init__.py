"""
Optimizers is much like optimizers in PyTorch,
but for the purpose of optimizing queries and search.
Each optimizer would perform a few steps that collectively
would guide the search towards the optimal trajectory.
"""

from .ax import AcquisitionFunc, AxServiceOptimizer, AxServiceParameter
from .brute import BruteForceOptimizer
from .evals import evaluate_corpus, evaluate_index, query_eval_func, search_eval_func
from .interfaces import Optimizer, QueryEvaluator, SearchEvaluator
from .random import RandomOptimizer
from .sklearn import (
    KMeansOptimizer,
    KMeansOptions,
    KMedoidsOptimizer,
    KMedoidsOptions,
    ScikitLearnOptimizer,
)
from .uniform import UniformOptimizer
from .utils import RemainingSteps

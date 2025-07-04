# Copyright (c) BoCoEL Authors - All Rights Reserved

"""
Language models (LMs) are models that are trained
to predict the next word in a sequence of words.
They are being evaluated in this project.

2 useful abstractions are defined in this module:

- ClassifierModel:
    A model that classifies text.
    Classifiers are used for predicting the class of a text.
    They do so by generating logits for each choice.
- GenerativeModel:
    A model that generates text.
    This is the LLM in common sense.
"""

from .huggingface import *
from .interfaces import *

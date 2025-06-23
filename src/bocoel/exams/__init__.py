# Copyright (c) BoCoEL Authors - All Rights Reserved

"""
The exams module provides the functionality to create and manage exams.
Here, an exam is used to measure how well the corpus or the model performs on a given task.

The module provides the following functionality:

- `Examinator`s are responsible for launch exams.
- `Exam`s are the tests that take in an accumulated history of model / corpus and returns a score.
- `Manager`s are responsible for managing results across runs.
"""

from .examinators import *
from .interfaces import *
from .managers import *
from .stats import *

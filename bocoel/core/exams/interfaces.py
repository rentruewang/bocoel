import abc
from collections import OrderedDict
from typing import Protocol

from numpy.typing import NDArray

from bocoel.corpora import StatefulIndex


class Exam(Protocol):
    @abc.abstractmethod
    def run(
        self, index: StatefulIndex, results: OrderedDict[int, float]
    ) -> NDArray: ...

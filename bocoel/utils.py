import abc
from typing import Any, Protocol

from typing_extensions import Self


class Configurable(Protocol):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: dict[str, Any]) -> Self:
        ...

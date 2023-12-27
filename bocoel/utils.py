from __future__ import annotations

import abc
from typing import Any, Dict, Protocol

from typing_extensions import Self


class Configurable(Protocol):
    @classmethod
    @abc.abstractmethod
    def from_config(cls, cfg: Dict[str, Any]) -> Self:
        ...

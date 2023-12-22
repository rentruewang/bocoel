from __future__ import annotations

from typing import Protocol


class Optimizer(Protocol):
    def __init__(self) -> None:
        super().__init__()

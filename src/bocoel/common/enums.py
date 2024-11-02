from enum import Enum
from typing import Self


class ItemNotFound(Exception):
    pass


class StrEnum(str, Enum):
    @classmethod
    def lookup(cls, name: str | Self) -> Self:
        if isinstance(name, cls):
            return name

        try:
            return cls[name]
        except KeyError:
            pass

        try:
            return cls(name)
        except ValueError:
            pass

        raise ItemNotFound(f"Item not found in enum. Must be one of {list(cls)}")

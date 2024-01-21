from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

from typing_extensions import Self

T = TypeVar("T")


class _BatchedGeneratorIterator(Iterator[list[T]]):
    def __init__(self, iterable: Iterable[T], batch_size: int, /) -> None:
        self._iterator = iter(iterable)
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[list[T]]:
        return self

    def __next__(self) -> list[T]:
        collected = []

        for _, data in zip(range(self._batch_size), self._iterator):
            collected.append(data)

        # Iterator is already empty.
        if not collected:
            raise StopIteration

        return collected


class BatchedGenerator(Generic[T]):
    def __init__(self, iterable: Iterable[T], batch_size: int) -> None:
        self._iterable = iterable
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[list[T]]:
        return _BatchedGeneratorIterator(self._iterable, self._batch_size)


class RemainingSteps:
    def __init__(self, count: int | float) -> None:
        self._count = count

    @property
    def count(self) -> int | float:
        return self._count

    def step(self, size: int = 1) -> None:
        self._count -= size

    @property
    def done(self) -> bool:
        return self._count <= 0

    @classmethod
    def infinite(cls) -> Self:
        return cls(count=float("inf"))

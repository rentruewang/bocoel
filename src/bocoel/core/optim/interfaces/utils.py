# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from collections.abc import Iterable, Iterator
from typing import Generic, TypeVar

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
    """
    A generator that yields batches of data from an iterable.
    """

    def __init__(self, iterable: Iterable[T], batch_size: int) -> None:
        """
        Parameters:
            iterable: The iterable to batch.
            batch_size: The size of the batches.
        """

        self._iterable = iterable
        self._batch_size = batch_size

    def __iter__(self) -> Iterator[list[T]]:
        return _BatchedGeneratorIterator(self._iterable, self._batch_size)


class RemainingSteps:
    """
    A simple counter that counts down the number of steps remaining.
    """

    def __init__(self, count: int | float) -> None:
        """
        Parameters:
            count: The number of steps remaining.
        """

        self._count = count

    @property
    def count(self) -> int | float:
        """
        The number of steps remaining.
        """

        return self._count

    def step(self, size: int = 1) -> None:
        """
        Perform a single step.

        Parameters:
            size: The number of steps to perform.
        """

        self._count -= size

    @property
    def done(self) -> bool:
        """
        Whether the number of steps is done.

        Returns:
            True if the number of steps is done, False otherwise.
        """

        return self._count <= 0

    @classmethod
    def infinite(cls) -> "RemainingSteps":
        """
        Create a counter that never ends.

        Returns:
            A counter that never ends.
        """

        return cls(count=float("inf"))

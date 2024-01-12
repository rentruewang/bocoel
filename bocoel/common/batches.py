from typing import Protocol


class Batched(Protocol):
    @property
    def batch_size(self) -> int:
        """
        The batch size used for processing.
        """

        ...

# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Protocol

from bocoel.models.adaptors.interfaces.adaptors import Adaptor
from bocoel.models.scores import Score


class BigBenchAdaptor(Adaptor, Protocol):
    _score_fn: Score

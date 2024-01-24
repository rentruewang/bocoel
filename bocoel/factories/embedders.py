from bocoel import Embedder, SbertEmbedder
from bocoel.common import StrEnum

from . import common


class EmbedderName(StrEnum):
    SBERT = "SBERT"


def embedder_factory(
    name: str | EmbedderName = EmbedderName.SBERT,
    /,
    *,
    model_name: str,
    device: str,
    batch_size: int,
) -> Embedder:
    if EmbedderName.lookup(name) is not EmbedderName.SBERT:
        raise ValueError(f"Unknown embedder name: {name}")

    return common.correct_kwargs(SbertEmbedder)(
        model_name=model_name, device=device, batch_size=batch_size
    )

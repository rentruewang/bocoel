from collections.abc import Mapping
from pathlib import Path
from typing import Any

import ujson as json
import yaml


def load(data: str | Path | Mapping[str, Any] | None, /) -> Mapping[str, Any]:
    if data is None:
        return {}

    if isinstance(data, str):
        data = Path(data)

    if isinstance(data, Path):
        data = parse_file(data)

    return data


def parse_file(file: str | Path) -> dict[str, Any]:
    file = str(file)
    if file.endswith(".yaml") or file.endswith(".yml"):
        with open(file) as f:
            return yaml.load(f)
    elif file.endswith(".json"):
        with open(file) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type: {file}")

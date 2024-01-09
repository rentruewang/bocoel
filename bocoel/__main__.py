from typing import Any

import fire
import ujson as json
import yaml


def main(file: str) -> None:
    data = parse_file(file)


def parse_file(file: str) -> dict[str, Any]:
    if file.endswith(".yaml") or file.endswith(".yml"):
        with open(file) as f:
            return yaml.load(f)
    elif file.endswith(".json"):
        with open(file) as f:
            return json.load(f)
    else:
        raise ValueError(f"Unknown file type: {file}")


if __name__ == "__main__":
    # Not a class. Google's standard public functions are all capitalized.
    fire.Fire(main)

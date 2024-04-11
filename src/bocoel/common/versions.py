from importlib import metadata


def version() -> str:
    return metadata.version("bocoel")

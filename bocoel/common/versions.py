# Copyright (c) RenChu Wang - All Rights Reserved

from importlib import metadata


def version() -> str:
    return metadata.version("bocoel")

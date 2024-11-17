# Copyright (c) 2024 RenChu Wang - All Rights Reserved

from typing import Any


def remove_base_suffix(self: Any, base_class: type[Any]) -> str:
    self_type_name: str = type(self).__name__
    base_class_name: str = base_class.__name__

    # In the case of the base class itself.
    if self_type_name == base_class_name:
        return base_class_name

    if not self_type_name.endswith(base_class_name):
        raise TypeError(
            f"Expected name `{self_type_name}` to end with `{base_class_name}`"
        )

    return self_type_name.replace(base_class_name, "")

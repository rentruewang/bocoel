from collections.abc import Sequence


def numeric_choices(question: str, choices: Sequence[str]) -> str:
    return f"{question}\nSelect from one of the following:\n" + "\n".join(
        f"{i}) {choice}" for i, choice in enumerate(choices, 1)
    )

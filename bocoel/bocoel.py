from bocoel.core import Optimizer, State


# FIXME: Should use a config dictionary to make it more configurable.
def bocoel(optimizer: Optimizer, iterations: int) -> list[State]:
    """
    This is the entry point fo the entire library.

    Parameters
    ----------

    # TODO: Parameters

    Returns
    -------

    The list of state changes given by the optimizer.
    """

    states: list[State] = []

    for _ in range(iterations):
        if optimizer.terminate:
            break

        states.extend(optimizer.step())

    return states

from bocoel.core import Optimizer


# FIXME: Should use a config dictionary to make it more configurable.
def bocoel(optimizer: Optimizer, iterations: int) -> dict[int, float]:
    """
    This is the entry point fo the entire library.

    Parameters
    ----------

    # TODO: Parameters

    Returns
    -------

    The list of state changes given by the optimizer.
    """

    states: dict[int, float] = {}

    for _ in range(iterations):
        try:
            states.update(optimizer.step())
        except StopIteration:
            break

    return states

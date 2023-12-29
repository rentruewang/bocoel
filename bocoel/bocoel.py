from bocoel.core import Core, State


# FIXME: Should use a config dictionary.
def bocoel(iterations: int, core: Core) -> list[State]:
    """
    This is the entry point fo the entire library.

    FIXME: Should make it more configurable (with the aid of factory functions).

    Parameters
    ----------

    # TODO: Parameters

    Returns
    -------

    The list of state changes given by the optimizer.
    """

    states = []

    for _ in range(iterations):
        state = core.step()
        states.append(state)

    return states

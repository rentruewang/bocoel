from bocoel.core import Core, State


# FIXME: Should use a config dictionary.
def bocoel(iterations: int, core: Core) -> list[State]:
    """
    This is the entry point fo the entire library.

    FIXME: Should make it more configurable (with the aid of factory functions).

    Parameters
    ----------

    `iteartions: int`
    FIXME: Should remove this in favor of config dictionary.

    `core: Core`
    The algorithms to use.
    A core contains both language model and corpus,
    and is responsible for finding the best language model.

    Returns
    -------

    The list of state changes given by the optimizer.
    """

    states = []

    for _ in range(iterations):
        state = core.optimize()
        states.append(state)

    return states

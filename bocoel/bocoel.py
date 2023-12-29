from bocoel.core import Core, State


# FIXME: Should use a config dictionary.
def bocoel(core: Core, iterations: int) -> list[State]:
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

    return core.run(iterations)

from torch import cuda


def devices() -> list[str]:
    """
    Avaialble devices that the embedders are supposed to run on.
    If CUDA is available, both CPU and CUDA are tested.
    """

    device_list = ["cpu"]

    if cuda.is_available():
        device_list.append("cuda")

    return device_list

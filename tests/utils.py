import faiss
from torch import cuda


def torch_devices() -> list[str]:
    """
    Avaialble devices that the embedders are supposed to run on.
    If CUDA is available, both CPU and CUDA are tested.
    """

    device_list = ["cpu"]

    if cuda.is_available():
        device_list.append("cuda")

    return device_list


def faiss_devices() -> list[str]:
    """
    Avaialble devices that the embedders are supposed to run on.
    If CUDA is available, both CPU and CUDA are tested.
    """

    device_list = ["cpu"]

    if faiss.get_num_gpus() > 0:
        device_list.append("cuda")

    return device_list

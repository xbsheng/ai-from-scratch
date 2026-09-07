import torch


def get_device() -> torch.device:
    """
    CUDA -> MPS(Apple Silicon) -> CPU
    """

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


DEVICE = get_device()

print("DEVICE", DEVICE)

"""CUDA device selection shared by pipeline and isolated workers."""

from __future__ import annotations

from fdanyone.errors import ConfigurationError


def select_cuda_device(device: str) -> tuple[str, int]:
    """Validate, select, and normalize one CUDA device."""

    import torch

    try:
        requested = torch.device(device)
    except (RuntimeError, TypeError, ValueError) as exc:
        raise ConfigurationError(f"Invalid CUDA device {device!r}.") from exc
    if requested.type != "cuda" or not torch.cuda.is_available():
        raise ConfigurationError(f"4DAnyone requires an available CUDA device, got {device!r}.")
    index = torch.cuda.current_device() if requested.index is None else requested.index
    if index < 0 or index >= torch.cuda.device_count():
        raise ConfigurationError(
            f"CUDA device index {index} is unavailable; visible device count is {torch.cuda.device_count()}."
        )
    torch.cuda.set_device(index)
    return f"cuda:{index}", index

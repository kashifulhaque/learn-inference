"""Lab 00 — confirm the GPU.

Fill in the two functions. This lab exists to prove the plumbing works before
anything harder depends on it.
"""

import torch

# Published A100 80GB peaks.
PEAK_BF16_TFLOPS = 312.0
PEAK_BANDWIDTH_GBS = 1935.0


def gpu_report() -> dict:
    """Return facts about the visible GPU.

    Returns:
        A dict with keys "name", "memory_gb", "sm_count", and "capability".
        "capability" is a string such as "8.0".
    """
    # TODO: read torch.cuda.get_device_properties(0) and fill this in.
    raise NotImplementedError


def ridge_point(tflops: float = PEAK_BF16_TFLOPS,
                bandwidth_gbs: float = PEAK_BANDWIDTH_GBS) -> float:
    """Return FLOPs per byte at which compute and bandwidth limits are equal.

    An operation below this number cannot saturate the tensor cores.
    """
    # TODO: convert both to base units and divide.
    raise NotImplementedError

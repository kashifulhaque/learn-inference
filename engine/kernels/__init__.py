"""Triton kernels that replace the PyTorch reference paths.

These import Triton lazily so the rest of the engine works on a machine without
a GPU. Each kernel has a `*_reference` twin in the layers package; the labs
check the kernel against it before timing it, because a fast wrong answer is
still wrong.
"""

HAS_TRITON = False
try:  # pragma: no cover - depends on the machine
    import triton  # noqa: F401

    HAS_TRITON = True
except ImportError:  # pragma: no cover
    pass

__all__ = ["HAS_TRITON"]

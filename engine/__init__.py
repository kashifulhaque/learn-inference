"""A teaching inference engine for Qwen3.5-family hybrid models.

Read the modules in this order; each one is the subject of a chapter:

    config       model geometry and the memory arithmetic that follows from it
    weights      safetensors loading
    layers/      RMSNorm, RoPE, GQA attention, gated-delta linear attention, MLP
    cache        the hybrid cache: growing KV plus fixed-size recurrent state
    model        the 64-layer stack
    sampling     turning logits into tokens
    scheduler    continuous batching
    kernels/     Triton kernels that replace the PyTorch reference paths
"""

__version__ = "0.1.0"

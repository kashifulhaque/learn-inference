"""Lab 01 — read the checkpoint index.

You are given the model's weight map and a table of tensor shapes. Answer
questions about the architecture without loading a single weight.

The harness calls your functions with:

    weight_map: dict[str, str]        tensor name -> shard filename
    shapes:     dict[str, list[int]]  tensor name -> shape
"""


def count_layers(weight_map: dict[str, str]) -> int:
    """Return how many decoder layers the checkpoint contains.

    Tensor names look like "model.language_model.layers.7.mlp.gate_proj.weight".
    """
    # TODO
    raise NotImplementedError


def classify_layers(weight_map: dict[str, str]) -> dict[int, str]:
    """Return {layer_index: "full_attention" or "linear_attention"}.

    A full-attention layer has tensors containing "self_attn". A linear-attention
    layer has tensors containing "linear_attn".
    """
    # TODO
    raise NotImplementedError


def largest_tensor(shapes: dict[str, list[int]]) -> tuple[str, int]:
    """Return the name and element count of the largest tensor."""
    # TODO
    raise NotImplementedError


def detects_output_gate(shapes: dict[str, list[int]], num_heads: int,
                        head_dim: int) -> bool:
    """Return True when a q_proj emits twice the width the heads need.

    A projection for `num_heads` heads of `head_dim` normally has
    num_heads * head_dim output rows. Twice that means the layer also emits a
    gate.
    """
    # TODO
    raise NotImplementedError

import re

LAYER_RE = re.compile(r"layers\.(\d+)\.")


def _layer_index(name: str) -> int | None:
    match = LAYER_RE.search(name)
    return int(match.group(1)) if match else None


def count_layers(weight_map: dict[str, str]) -> int:
    indices = {i for name in weight_map if (i := _layer_index(name)) is not None}
    return len(indices)


def classify_layers(weight_map: dict[str, str]) -> dict[int, str]:
    kinds: dict[int, str] = {}
    for name in weight_map:
        index = _layer_index(name)
        if index is None:
            continue
        if "self_attn" in name:
            kinds[index] = "full_attention"
        elif "linear_attn" in name and index not in kinds:
            kinds[index] = "linear_attention"
    return kinds


def largest_tensor(shapes: dict[str, list[int]]) -> tuple[str, int]:
    best_name, best_size = "", -1
    for name, shape in shapes.items():
        size = 1
        for dim in shape:
            size *= dim
        if size > best_size:
            best_name, best_size = name, size
    return best_name, best_size


def detects_output_gate(shapes: dict[str, list[int]], num_heads: int,
                        head_dim: int) -> bool:
    expected = num_heads * head_dim
    for name, shape in shapes.items():
        if name.endswith("q_proj.weight"):
            return shape[0] == 2 * expected
    return False

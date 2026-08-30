"""Checks lab 01 against a synthetic index shaped like the real checkpoint."""

from lab_common import Checks

NUM_LAYERS = 64
INTERVAL = 4
HIDDEN = 5120
HEADS, HEAD_DIM, KV_HEADS = 24, 256, 4
INTERMEDIATE = 17408
VOCAB = 248320


def _build_index():
    weight_map, shapes = {}, {}

    def add(name, shape):
        weight_map[name] = f"model-{(len(weight_map) // 60) + 1:05d}-of-00018.safetensors"
        shapes[name] = shape

    add("model.language_model.embed_tokens.weight", [VOCAB, HIDDEN])
    add("lm_head.weight", [VOCAB, HIDDEN])
    add("model.language_model.norm.weight", [HIDDEN])

    for i in range(NUM_LAYERS):
        prefix = f"model.language_model.layers.{i}."
        add(prefix + "input_layernorm.weight", [HIDDEN])
        add(prefix + "post_attention_layernorm.weight", [HIDDEN])
        add(prefix + "mlp.gate_proj.weight", [INTERMEDIATE, HIDDEN])
        add(prefix + "mlp.up_proj.weight", [INTERMEDIATE, HIDDEN])
        add(prefix + "mlp.down_proj.weight", [HIDDEN, INTERMEDIATE])
        if (i + 1) % INTERVAL == 0:
            # The q_proj is deliberately double width: this model gates its
            # attention output.
            add(prefix + "self_attn.q_proj.weight", [2 * HEADS * HEAD_DIM, HIDDEN])
            add(prefix + "self_attn.k_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN])
            add(prefix + "self_attn.v_proj.weight", [KV_HEADS * HEAD_DIM, HIDDEN])
            add(prefix + "self_attn.o_proj.weight", [HIDDEN, HEADS * HEAD_DIM])
        else:
            add(prefix + "linear_attn.in_proj_qkvz.weight", [16384, HIDDEN])
            add(prefix + "linear_attn.in_proj_ba.weight", [96, HIDDEN])
            add(prefix + "linear_attn.conv1d.weight", [10240, 1, 4])
            add(prefix + "linear_attn.A_log", [48])
            add(prefix + "linear_attn.dt_bias", [48])
            add(prefix + "linear_attn.out_proj.weight", [HIDDEN, 6144])
    return weight_map, shapes


def run(submission):
    c = Checks()
    if not c.require(
        submission, "count_layers", "classify_layers", "largest_tensor",
        "detects_output_gate",
    ):
        return c.finish()

    weight_map, shapes = _build_index()

    c.check(
        "count_layers finds 64 layers",
        lambda: submission.count_layers(weight_map) == NUM_LAYERS,
        f"got {submission.count_layers(weight_map)}",
    )

    kinds = submission.classify_layers(weight_map)
    full = sorted(i for i, kind in kinds.items() if kind == "full_attention")
    linear = sorted(i for i, kind in kinds.items() if kind == "linear_attention")

    c.check("every layer is classified", lambda: len(kinds) == NUM_LAYERS,
            f"got {len(kinds)}")
    c.check("16 layers are full attention", lambda: len(full) == 16, f"got {len(full)}")
    c.check("48 layers are linear attention", lambda: len(linear) == 48,
            f"got {len(linear)}")
    c.check(
        "full attention sits on every fourth layer",
        lambda: full == [i for i in range(NUM_LAYERS) if (i + 1) % INTERVAL == 0],
        f"first few: {full[:5]}",
    )

    name, size = submission.largest_tensor(shapes)
    c.check(
        "the largest tensor is an embedding-sized matrix",
        lambda: size == VOCAB * HIDDEN,
        f"{name} with {size:,} elements",
    )

    c.check(
        "the output gate is detected from the q_proj shape",
        lambda: submission.detects_output_gate(shapes, HEADS, HEAD_DIM) is True,
    )
    c.check(
        "no gate is reported when the q_proj is normal width",
        lambda: submission.detects_output_gate(
            {"model.layers.0.self_attn.q_proj.weight": [HEADS * HEAD_DIM, HIDDEN]},
            HEADS, HEAD_DIM,
        ) is False,
    )

    c.metric("total_tensors", len(weight_map))
    c.metric("largest_tensor_params", size)
    c.metric("full_attention_layers", len(full))
    return c.finish()

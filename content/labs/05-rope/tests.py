import torch
from lab_common import Checks, close, pick_device

HEAD_DIM = 256
ROTARY_DIM = 64          # partial_rotary_factor 0.25
THETA = 10_000_000.0


def run(submission):
    c = Checks()
    if not c.require(submission, "build_rope_cache", "rotate_half",
                     "apply_rotary_partial"):
        return c.finish()

    device = pick_device()
    torch.manual_seed(0)

    cos, sin = submission.build_rope_cache(ROTARY_DIM, 512, THETA, device=device)
    c.check(
        "the tables have shape (positions, rotary_dim // 2)",
        lambda: tuple(cos.shape) == (512, ROTARY_DIM // 2),
        f"got {tuple(cos.shape)}",
    )
    c.check(
        "position 0 has no rotation",
        lambda: close(cos[0], torch.ones(ROTARY_DIM // 2, device=device), 1e-6)[0]
        and close(sin[0], torch.zeros(ROTARY_DIM // 2, device=device), 1e-6)[0],
    )
    c.check(
        "cos^2 + sin^2 is 1 everywhere",
        lambda: close(cos**2 + sin**2, torch.ones_like(cos), 1e-5),
    )
    c.check(
        "the first frequency rotates fastest",
        lambda: (
            torch.atan2(sin[1, 0], cos[1, 0]).abs()
            > torch.atan2(sin[1, -1], cos[1, -1]).abs()
        ).item(),
    )

    x = torch.randn(1, 2, 4, 8, device=device)
    half = submission.rotate_half(x)
    c.check(
        "rotate_half maps (x1, x2) to (-x2, x1)",
        lambda: close(half, torch.cat((-x[..., 4:], x[..., :4]), dim=-1), 1e-6),
    )

    # The channels past rotary_dim must survive untouched.
    q = torch.randn(2, 3, 16, HEAD_DIM, device=device)
    cos_g, sin_g = cos[:16], sin[:16]
    out = submission.apply_rotary_partial(q, cos_g, sin_g, ROTARY_DIM)
    c.check("the output keeps its shape", lambda: out.shape == q.shape,
            f"got {tuple(out.shape)}")
    c.check(
        "channels past the rotary boundary are unchanged",
        lambda: close(out[..., ROTARY_DIM:], q[..., ROTARY_DIM:], 1e-6),
    )
    c.check(
        "channels before the boundary do change",
        lambda: (out[..., :ROTARY_DIM] - q[..., :ROTARY_DIM]).abs().max().item() > 1e-3,
    )
    c.check(
        "the rotation preserves the norm of the rotated part",
        lambda: close(
            out[..., :ROTARY_DIM].norm(dim=-1), q[..., :ROTARY_DIM].norm(dim=-1), 1e-4
        ),
    )

    # The property the whole scheme exists for: the score depends on m - n only.
    gen = torch.Generator(device=device).manual_seed(7)
    base_q = torch.randn(1, 1, 1, HEAD_DIM, device=device, generator=gen)
    base_k = torch.randn(1, 1, 1, HEAD_DIM, device=device, generator=gen)

    def rotated_dot(m: int, n: int) -> float:
        qm = submission.apply_rotary_partial(
            base_q, cos[m : m + 1], sin[m : m + 1], ROTARY_DIM)
        kn = submission.apply_rotary_partial(
            base_k, cos[n : n + 1], sin[n : n + 1], ROTARY_DIM)
        return (qm * kn).sum().item()

    pairs = [(10, 5), (20, 15), (100, 95), (300, 295)]
    scores = [rotated_dot(m, n) for m, n in pairs]
    spread = max(scores) - min(scores)
    scale = max(abs(s) for s in scores) + 1e-9
    relative = spread / scale

    c.check(
        "the score depends only on the gap between positions",
        lambda: relative < 1e-4,
        f"relative spread across four pairs with gap 5: {relative:.2e}",
    )
    c.check(
        "a different gap gives a different score",
        lambda: abs(rotated_dot(10, 5) - rotated_dot(10, 0)) > 1e-4,
    )

    c.metric("rotary_dim", ROTARY_DIM)
    c.metric("relative_error", float(f"{relative:.3e}"))
    return c.finish()

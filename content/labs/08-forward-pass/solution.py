import torch


def logit_metrics(mine: torch.Tensor, reference: torch.Tensor) -> dict:
    mine32 = mine.float()
    ref32 = reference.float()

    flat_mine = mine32.flatten()
    flat_ref = ref32.flatten()
    stacked = torch.stack([flat_mine, flat_ref])
    correlation = torch.corrcoef(stacked)[0, 1].item()

    log_p = torch.log_softmax(ref32, dim=-1)
    log_q = torch.log_softmax(mine32, dim=-1)
    kl = (log_p.exp() * (log_p - log_q)).sum(dim=-1).mean().item()

    return {
        "max_abs_diff": (mine32 - ref32).abs().max().item(),
        "correlation": correlation,
        "top1_agreement": (
            mine32.argmax(-1) == ref32.argmax(-1)
        ).float().mean().item(),
        "mean_kl": kl,
    }


def first_divergent_layer(mine, reference, tol: float = 1e-2):
    for index, (a, b) in enumerate(zip(mine, reference)):
        if (a.float() - b.float()).abs().max().item() > tol:
            return index
    return None


def prefill_decode_equivalence(model, input_ids, prefill_len: int, make_cache) -> float:
    with torch.no_grad():
        reference = model(input_ids)

        cache = make_cache()
        model(input_ids[:, :prefill_len], cache=cache, last_token_only=True)

        worst = 0.0
        total = input_ids.shape[1]
        for position in range(prefill_len, total):
            logits = model(
                input_ids[:, position : position + 1], cache=cache, last_token_only=True
            )
            diff = (logits[:, -1] - reference[:, position]).abs().max().item()
            worst = max(worst, diff)
    return worst

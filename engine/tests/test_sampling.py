"""Top-k / log-softmax and the sampler."""

import torch

from q27lib import call

dev = "cuda"
V = 248320


def sample(logits, temperature, top_k, top_p, u, out, min_p=0.0, counts=None, presence=0.0, frequency=0.0):
    call("q27t_sample", logits, logits.shape[0], V, temperature, top_k, top_p, u, out, min_p,
         0 if counts is None else counts, presence, frequency)


def test_logsoftmax_topk():
    T, k = 5, 20
    logits = torch.randn(T, V, device=dev) * 4
    logits[1, 100:110] = 30.0  # ties: lower index first
    ids, logp = torch.empty(T, k, dtype=torch.int32, device=dev), torch.empty(T, k, device=dev)
    call("q27t_logsoftmax_topk", logits, T, V, k, ids, logp)
    lp = torch.log_softmax(logits.double(), -1)
    ref_v, ref_i = lp.topk(k, -1)
    torch.testing.assert_close(logp.double(), ref_v, rtol=0, atol=1e-5)
    assert torch.equal(ids[1, :10].cpu(), torch.arange(100, 110, dtype=torch.int32))
    assert torch.equal(ids[[0, 2, 3, 4]].long(), ref_i[[0, 2, 3, 4]])


def test_sample_greedy_and_distribution():
    logits = torch.full((1, V), -30.0, device=dev)
    logits[0, [5, 7, 9]] = torch.tensor([2.0, 1.0, 0.0], device=dev)
    out = torch.empty(1, dtype=torch.int32, device=dev)
    u = torch.empty(1, device=dev)
    sample(logits, 0.0, 20, 1.0, u, out)
    assert out.item() == 5
    # temperature 1, top_k 20, top_p 1: frequencies follow softmax over {5, 7, 9}.
    counts = {5: 0, 7: 0, 9: 0}
    n = 3000
    for x in torch.rand(n).tolist():
        u.fill_(x)
        sample(logits, 1.0, 20, 1.0, u, out)
        counts[out.item()] += 1
    p = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), 0)
    for (tok, c), pt in zip(counts.items(), p.tolist()):
        assert abs(c / n - pt) < 0.03, (tok, c / n, pt)
    # top_p 0.5 keeps only the most likely token (p = 0.665).
    u.fill_(0.99)
    sample(logits, 1.0, 20, 0.5, u, out)
    assert out.item() == 5


def test_sample_topk_matches_full_sort():
    """The two-stage top-k against torch over a row with ties and segment-boundary candidates."""
    logits = torch.randn(1, V, device=dev)  # max ~5 over 248K samples
    logits[0, 2040:2056] = 12.0  # ties across the first segment boundary
    logits[0, V - 5:] = 11.0     # the partial last segment
    out = torch.empty(1, dtype=torch.int32, device=dev)
    u = torch.empty(1, device=dev)
    sample(logits, 0.0, 20, 1.0, u, out)
    assert out.item() == 2040
    # With temperature 1e-3 and top_k 64 the pick among the 16 tied tokens follows u.
    picks = set()
    for x in (0.01, 0.3, 0.6, 0.99):
        u.fill_(x)
        sample(logits, 1e-3, 64, 1.0, u, out)
        picks.add(out.item())
    assert picks <= set(range(2040, 2056)) and len(picks) == 4


def test_sample_min_p():
    logits = torch.full((1, V), -30.0, device=dev)
    logits[0, [5, 7, 9]] = torch.tensor([2.0, 1.0, 0.0], device=dev)  # p / p_max = 1, 0.37, 0.14
    out = torch.empty(1, dtype=torch.int32, device=dev)
    u = torch.empty(1, device=dev)
    seen = set()
    for x in (0.0, 0.5, 0.75, 0.999):
        u.fill_(x)
        sample(logits, 1.0, 20, 1.0, u, out, min_p=0.2)
        seen.add(out.item())
    assert seen == {5, 7}
    # Within what min_p keeps, the frequencies follow softmax over {5, 7}.
    n, hits = 2000, 0
    for x in torch.rand(n).tolist():
        u.fill_(x)
        sample(logits, 1.0, 20, 1.0, u, out, min_p=0.2)
        hits += out.item() == 5
    assert abs(hits / n - 1 / (1 + torch.e ** -1)) < 0.03


def test_sample_penalties():
    """OpenAI's presence / frequency penalties on the logits, the top k taken after them, and the count kept."""
    logits = torch.randn(1, V, device=dev)
    top = torch.arange(100, 164, device=dev)
    logits[0, top] = torch.linspace(20.0, 10.0, 64, device=dev)  # the 64 highest
    counts = torch.zeros(V, dtype=torch.int32, device=dev)
    counts[100], counts[101], counts[102] = 1, 3, 2
    out = torch.empty(1, dtype=torch.int32, device=dev)
    u = torch.empty(1, device=dev)
    pres, freq = 1.5, 0.75
    pen = logits[0].double().clone()
    c = counts.double()
    pen -= pres * (c > 0) + freq * c
    ref = pen[top].sort(descending=True)
    # greedy: the best after the penalties (and the token counted)
    sample(logits, 0.0, 20, 1.0, u, out, counts=counts, presence=pres, frequency=freq)
    assert out.item() == top[ref.indices[0]].item()
    assert counts[out.item()].item() == 1
    counts[out.item()] = 0
    # top_k 4 after the penalties, temperature 1: the frequencies follow softmax over those 4.
    keep = top[ref.indices[:4]].tolist()
    p = torch.softmax(ref.values[:4], 0).tolist()
    n, seen = 3000, {t: 0 for t in keep}
    for x in torch.rand(n).tolist():
        u.fill_(x)
        sample(logits, 1.0, 4, 1.0, u, out, counts=counts, presence=pres, frequency=freq)
        t = out.item()
        assert t in seen, t
        seen[t] += 1
        counts[t] -= 1  # (undo the count)
    for t, pt in zip(keep, p):
        assert abs(seen[t] / n - pt) < 0.03, (t, seen[t] / n, pt)
    assert counts[100].item() == 1 and counts[101].item() == 3 and counts[102].item() == 2

"""Top-k / log-softmax and the sampler."""

import torch

from q27lib import call

dev = "cuda"
V = 248320


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
    call("q27t_sample", logits, 1, V, 0.0, 20, 1.0, u, out)
    assert out.item() == 5
    # temperature 1, top_k 20, top_p 1: frequencies follow softmax over {5, 7, 9}.
    counts = {5: 0, 7: 0, 9: 0}
    n = 3000
    for x in torch.rand(n).tolist():
        u.fill_(x)
        call("q27t_sample", logits, 1, V, 1.0, 20, 1.0, u, out)
        counts[out.item()] += 1
    p = torch.softmax(torch.tensor([2.0, 1.0, 0.0]), 0)
    for (tok, c), pt in zip(counts.items(), p.tolist()):
        assert abs(c / n - pt) < 0.03, (tok, c / n, pt)
    # top_p 0.5 keeps only the most likely token (p = 0.665).
    u.fill_(0.99)
    call("q27t_sample", logits, 1, V, 1.0, 20, 0.5, u, out)
    assert out.item() == 5


def test_sample_topk_matches_full_sort():
    """The two-stage top-k against torch over a row with ties and segment-boundary candidates."""
    logits = torch.randn(1, V, device=dev)  # max ~5 over 248K samples
    logits[0, 2040:2056] = 12.0  # ties across the first segment boundary
    logits[0, V - 5:] = 11.0     # the partial last segment
    out = torch.empty(1, dtype=torch.int32, device=dev)
    u = torch.empty(1, device=dev)
    call("q27t_sample", logits, 1, V, 0.0, 20, 1.0, u, out)
    assert out.item() == 2040
    # With temperature 1e-3 and top_k 64 the pick among the 16 tied tokens follows u.
    picks = set()
    for x in (0.01, 0.3, 0.6, 0.99):
        u.fill_(x)
        call("q27t_sample", logits, 1, V, 1e-3, 64, 1.0, u, out)
        picks.add(out.item())
    assert picks <= set(range(2040, 2056)) and len(picks) == 4

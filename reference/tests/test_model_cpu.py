"""Whole-model sanity check on the CPU (slow: streams all 64 layers from the checkpoint).

    pytest -m slow tests/test_model_cpu.py
"""

import pytest
import torch
from transformers import AutoTokenizer

from qwen27b.model import Qwen38Reference, SeqState
from qwen27b.weights import TARGET, Checkpoint, snapshot


def _kl(p_logits, q_logits):
    """KL(q || p) of two logit vectors."""
    p, q = p_logits.log_softmax(-1), q_logits.log_softmax(-1)
    return (q.exp() * (q - p)).sum().item()


def _step_vs_full(ref, ids):
    """Prefill `ids`, decode the greedy next token through the recurrent GDN path, and re-prefill the
    longer prompt with the chunked path. Returns (prefill logits, step logits, full logits)."""
    st = SeqState()
    logits = ref.forward(ids, st)
    nxt = logits[0].argmax().view(1)
    ref.gdn = "recurrent"
    step = ref.forward(nxt, st)[0]
    ref.gdn = "chunked"
    full = ref.forward(torch.cat((ids, nxt)), SeqState())[0]
    return logits, step, full


@pytest.mark.slow
def test_next_token_and_incremental_decode():
    path = snapshot(*TARGET)
    tok = AutoTokenizer.from_pretrained(path)
    ckpt = Checkpoint(path)
    ids = torch.tensor(tok("The capital of France is").input_ids)

    # Without activation quantization the state carried between calls (GDN state, conv state, KV) must make
    # one decode step equal a fresh prefill of the longer prompt (measured: rel 4.5e-7).
    ref = Qwen38Reference(ckpt, act_quant=False, kv_dtype="fp32")
    logits, step, full = _step_vs_full(ref, ids)
    assert tok.decode(logits[0].argmax()).strip() == "Paris"
    assert torch.allclose(step, full, rtol=0, atol=1e-5 * full.abs().max()), (step - full).abs().max()

    # With W4A4 / W8A8 activation quantization, the ~1e-6 difference between a T=1 and a T=6 matmul flips
    # some FP4 / FP8 roundings and grows over 64 layers (measured: max |d| 7% of max |logit|, KL 0.005;
    # identical for the chunked and recurrent step). Only the top token and a loose KL bound hold.
    ref = Qwen38Reference(ckpt)
    logits, step, full = _step_vs_full(ref, ids)
    assert tok.decode(logits[0].argmax()).strip() == "Paris"
    assert step.argmax() == full.argmax()
    assert _kl(step, full) < 0.05

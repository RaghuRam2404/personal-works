# code is AI generated
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attention import CausalSelfAttention, multihead_loop, vector_attention  # noqa: E402


# Part 8 — One-head equivalence test
# Point: with n_head=1 (D == C), dropout=0, eval mode and c_proj = identity, the module
# must equal Session 12's vector_attention on the output and on all gradients.
def test_one_head_equivalence():
    """Part 8: n_head=1, dropout=0, eval, c_proj=identity -> matches vector_attention
    on output and on gradients for x and each projection weight."""
    torch.manual_seed(0)
    B, T, C = 3, 5, 6  # all different; n_head=1 so D == C

    m = CausalSelfAttention(C, 1, block_size=8, dropout=0.0).eval()
    with torch.no_grad():
        m.c_proj.weight.copy_(torch.eye(C, dtype=torch.float64))

    # Module weights are [out, in]; vector_attention also takes [d, C] and does x @ W.T,
    # so the weights are copied as-is (no transpose needed in this repo's convention).
    Wq = m.Wq.weight.detach().clone().requires_grad_()
    Wk = m.Wk.weight.detach().clone().requires_grad_()
    Wv = m.Wv.weight.detach().clone().requires_grad_()

    x_mod = torch.randn(B, T, C, dtype=torch.float64, requires_grad=True)
    x_ref = x_mod.detach().clone().requires_grad_()

    out_mod = m(x_mod)
    out_ref, w_ref = vector_attention(x_ref, Wq, Wk, Wv)

    torch.testing.assert_close(out_mod, out_ref)
    torch.testing.assert_close(m.last_weights[:, 0], w_ref)
    torch.testing.assert_close(m.last_y, out_ref)  # identity c_proj, so pre-proj y == out

    # non-trivial upstream gradient so the check isn't just all-ones
    R = torch.randn_like(out_ref)
    (out_mod * R).sum().backward()
    (out_ref * R).sum().backward()

    torch.testing.assert_close(x_mod.grad, x_ref.grad)
    torch.testing.assert_close(m.Wq.weight.grad, Wq.grad)
    torch.testing.assert_close(m.Wk.weight.grad, Wk.grad)
    torch.testing.assert_close(m.Wv.weight.grad, Wv.grad)


# Part 9 — Module versus loop-over-heads test
# Point: with n_head=2 and 3, the heads differ, and the module's pre-projection y and
# pre-dropout weights match multihead_loop, so each head keeps its own feature subspace.
@pytest.mark.parametrize("n_head", [2, 3])
def test_module_matches_multihead_loop(n_head):
    """Part 9: pre-projection y and pre-dropout weights match multihead_loop,
    so heads stay in separate feature subspaces before c_proj."""
    torch.manual_seed(1)
    B, T, C = 2, 5, 12  # divisible by both 2 and 3

    m = CausalSelfAttention(C, n_head, block_size=8, dropout=0.0).eval()
    with torch.no_grad():
        m.c_proj.weight.copy_(torch.eye(C, dtype=torch.float64))

    x = torch.randn(B, T, C, dtype=torch.float64)
    out = m(x)

    # heads must actually differ
    for h in range(1, n_head):
        assert not torch.allclose(m.last_weights[:, 0], m.last_weights[:, h])

    # multihead_loop slices rows [h*D:(h+1)*D] of [C, C] weights, same layout as nn.Linear.weight
    y_ref, w_ref = multihead_loop(
        x, m.Wq.weight.detach(), m.Wk.weight.detach(), m.Wv.weight.detach(), n_head
    )

    assert m.last_weights.shape == (B, n_head, T, T)
    torch.testing.assert_close(m.last_weights, w_ref.to(m.last_weights.dtype))
    torch.testing.assert_close(m.last_y, y_ref.to(m.last_y.dtype))
    torch.testing.assert_close(out, m.last_y)  # identity c_proj


# ---------------------------------------------------------------- Part 10
# all-different sizes: B=2, T=5, H=3, D=4 -> C=12
B10, T10, H10, D10 = 2, 5, 3, 4
C10 = H10 * D10
BLOCK10 = 8


def _make(dropout=0.0, train=False):
    torch.manual_seed(3)
    m = CausalSelfAttention(C10, H10, block_size=BLOCK10, dropout=dropout)
    return m.train() if train else m.eval()


def _prefix_perturbation(m, seed=None, t0=3):
    """Run x and x-with-perturbed-future; return (out, out_perturbed, t0)."""
    g = torch.Generator().manual_seed(11)
    x = torch.randn(B10, T10, C10, dtype=torch.float64, generator=g)
    xp = x.clone()
    xp[:, t0:] += torch.randn_like(xp[:, t0:]) * 5.0
    if seed is not None:
        torch.manual_seed(seed)  # same dropout masks for both runs
    out = m(x)
    if seed is not None:
        torch.manual_seed(seed)
    outp = m(xp)
    return out, outp, t0


# Part 10 — Causality and dropout tests
# Point: perturbing future inputs does not change the prefix output (eval mode).
def test_prefix_unchanged_by_future_eval():
    out, outp, t0 = _prefix_perturbation(_make())
    torch.testing.assert_close(out[:, :t0], outp[:, :t0])
    assert not torch.allclose(out[:, t0:], outp[:, t0:])  # sanity: future does change


# Part 10 — Causality and dropout tests
# Point: every head has zero probability above the diagonal.
def test_zero_above_diagonal_every_head():
    m = _make()
    m(torch.randn(B10, T10, C10, dtype=torch.float64))
    w = m.last_weights
    assert w.shape == (B10, H10, T10, T10)
    above = torch.triu(torch.ones(T10, T10, dtype=torch.bool), diagonal=1)
    assert (w[..., above] == 0).all()


# Part 10 — Causality and dropout tests
# Point: attention rows sum to 1 before dropout (after dropout they no longer do).
def test_rows_sum_to_one_before_dropout():
    # even in train mode with heavy dropout, last_weights is the pre-dropout copy
    m = _make(dropout=0.5, train=True)
    m(torch.randn(B10, T10, C10, dtype=torch.float64))
    sums = m.last_weights.sum(dim=-1)
    torch.testing.assert_close(sums, torch.ones_like(sums))


# Part 10 — Causality and dropout tests
# Point: eval mode disables dropout and is deterministic: two calls give identical outputs.
def test_eval_is_deterministic():
    m = _make(dropout=0.5)  # eval(): dropout must be disabled
    x = torch.randn(B10, T10, C10, dtype=torch.float64)
    torch.testing.assert_close(m(x), m(x), rtol=0, atol=0)


# Part 10 — Causality and dropout tests
# Point: training mode with dropout=0.5 varies between calls but stays causal
# (seed before each prefix-perturbation run so both draw the same dropout masks).
def test_train_varies_but_stays_causal():
    m = _make(dropout=0.5, train=True)
    x = torch.randn(B10, T10, C10, dtype=torch.float64)
    assert not torch.allclose(m(x), m(x))  # different dropout masks each call

    # same seed before each run -> same dropout masks -> prefix must be identical
    out, outp, t0 = _prefix_perturbation(m, seed=123)
    torch.testing.assert_close(out[:, :t0], outp[:, :t0])

    # pre-dropout weights are still causal in train mode
    above = torch.triu(torch.ones(T10, T10, dtype=torch.bool), diagonal=1)
    assert (m.last_weights[..., above] == 0).all()


# Part 10 — Causality and dropout tests
# Point: T > block_size raises an error.
def test_T_greater_than_block_size_raises():
    m = _make()
    with pytest.raises(AssertionError):
        m(torch.randn(B10, BLOCK10 + 1, C10, dtype=torch.float64))


# Part 10 — Causality and dropout tests (extra, not in the doc's list)
# Point: T < block_size works, which checks the mask[:T, :T] slice.
def test_shorter_than_block_size_works():
    m = _make()
    out = m(torch.randn(B10, 2, C10, dtype=torch.float64))
    assert out.shape == (B10, 2, C10)


# Part 10 — Causality and dropout tests
# Point: the mask buffer appears in named_buffers() and not in parameters().
def test_mask_is_buffer_not_parameter():
    m = _make()
    assert "mask" in dict(m.named_buffers())
    assert "mask" not in dict(m.named_parameters())
    assert "mask" not in m.state_dict()  # persist=False default
    assert "mask" in CausalSelfAttention(C10, H10, BLOCK10, 0.0, persist=True).state_dict()


if __name__ == "__main__":
    test_one_head_equivalence()
    print("part 8 ok")
    for h in (2, 3):
        test_module_matches_multihead_loop(h)
    print("part 9 ok")

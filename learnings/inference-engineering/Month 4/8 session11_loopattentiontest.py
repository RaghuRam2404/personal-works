import pytest
from gpt.attention import loop_attention
import torch

DT = torch.float64
ATOL = 1e-10


@pytest.fixture(params=range(5), ids=lambda s: f"seed{s}")
def get_default_inputs(request):
    # all-different sizes so an axis mix-up raises instead of passing silently
    B, T, D = 2, 5, 6
    dq, dk, dv = 4, 4, 3
    g = torch.Generator().manual_seed(request.param)
    torch.manual_seed(request.param)  # for the torch.randn calls inside tests
    x = torch.randn(size=(B, T, D), generator=g, dtype=DT)
    Wq = torch.randn(size=(dq, D), generator=g, dtype=DT)
    Wk = torch.randn(size=(dk, D), generator=g, dtype=DT)
    Wv = torch.randn(size=(dv, D), generator=g, dtype=DT)
    return B, T, D, dq, dk, dv, x, Wq, Wk, Wv


def test_diff_shapes():
    shapes = [
        # B, T, D, dq, dv
        (1, 3, 4, 5, 5),
        (2, 5, 6, 4, 3),
        (3, 3, 3, 3, 3),
        (1, 2, 3, 4, 5),
    ]
    for B, T, D, dq, dv in shapes:
        dk = dq
        x = torch.randn(size=(B, T, D), dtype=DT)
        Wq = torch.randn(size=(dq, D), dtype=DT)
        Wk = torch.randn(size=(dk, D), dtype=DT)
        Wv = torch.randn(size=(dv, D), dtype=DT)
        out, weights = loop_attention(x, Wq, Wk, Wv)
        assert out.shape == (B, T, dv)
        assert weights.shape == (B, T, T)


def test_float64_preserved_and_precise(get_default_inputs):
    *_, x, Wq, Wk, Wv = get_default_inputs
    out, w = loop_attention(x, Wq, Wk, Wv)
    assert out.dtype == torch.float64 and w.dtype == torch.float64

    # float32 run must differ by ~1e-7; a difference of 0 means a silent upcast
    out32, _ = loop_attention(x.float(), Wq.float(), Wk.float(), Wv.float())
    assert out32.dtype == torch.float32
    err = (out - out32.double()).abs().max()
    assert 1e-12 < err < 1e-4


def test_attention_row_sum_is_1(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    out, weights = loop_attention(x, Wq, Wk, Wv)
    assert torch.allclose(weights.sum(dim=2), torch.ones(B, T, dtype=DT), atol=ATOL)


def test_weights_is_lower_triangular(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    out, weights = loop_attention(x, Wq, Wk, Wv)
    assert (torch.triu(weights, diagonal=1) == 0).all()  # exactly zero


def test_future_token_wont_affect_past_tokens(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    out, weights = loop_attention(x, Wq, Wk, Wv)
    for future_i in range(1, T):
        xnew = x.clone()
        # independent random values for every future token
        xnew[:, future_i:, :] = torch.randn_like(xnew[:, future_i:, :])
        out_new, weights_new = loop_attention(xnew, Wq, Wk, Wv)
        assert torch.allclose(out[:, :future_i], out_new[:, :future_i], atol=ATOL)
        assert torch.allclose(weights[:, :future_i], weights_new[:, :future_i], atol=ATOL)
        assert not torch.allclose(out[:, future_i:], out_new[:, future_i:])
        assert not torch.allclose(weights[:, future_i:], weights_new[:, future_i:])


def test_wq0_same_prob(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    Wq0 = torch.zeros(size=(dq, D), dtype=DT)
    out0, weights0 = loop_attention(x, Wq0, Wk, Wv)
    for b in range(B):
        for i in range(T):
            expected = torch.full((i + 1,), 1 / (i + 1), dtype=DT)
            assert torch.allclose(weights0[b, i, : i + 1], expected, atol=ATOL)


def test_single_token_same_as_v():
    B, T, D = 2, 1, 4
    dq, dk, dv = 5, 5, 6
    x1 = torch.randn(size=(B, T, D), dtype=DT)
    Wq = torch.randn(size=(dq, D), dtype=DT)
    Wk = torch.randn(size=(dk, D), dtype=DT)
    Wv = torch.randn(size=(dv, D), dtype=DT)
    out1, weights1 = loop_attention(x1, Wq, Wk, Wv)
    assert torch.allclose(out1, x1 @ Wv.T, atol=ATOL)


def test_non_blowup_for_higher_wq(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    outE, weightsE = loop_attention(x, Wq * 1e4, Wk, Wv)
    assert torch.isfinite(outE).all()
    assert torch.isfinite(weightsE).all()
    assert torch.allclose(weightsE.sum(dim=2), torch.ones(B, T, dtype=DT), atol=ATOL)


def test_grad(get_default_inputs):
    B, T, D, dq, dk, dv, x, Wq, Wk, Wv = get_default_inputs
    for W in (Wq, Wk, Wv):
        W.requires_grad = True
    out, weights = loop_attention(x, Wq, Wk, Wv)
    out.sum().backward()
    for W in (Wq, Wk, Wv):
        assert W.grad is not None
        assert torch.isfinite(W.grad).all()
        assert W.grad.abs().sum() > 0


def test_part5_hand_example():
    # expected values come from the hand calculation / curriculum table, not from the code
    x = torch.tensor([[[1., 0.], [0., 1.], [1., 1.]]], dtype=DT)
    I = torch.eye(2, dtype=DT)
    out, w = loop_attention(x, I, I, I)

    def t(*v):
        return torch.tensor(v, dtype=DT)

    assert torch.allclose(w[0, 0, :1], t(1.0), atol=1e-4)
    assert torch.allclose(out[0, 0], t(1., 0.), atol=1e-4)
    assert torch.allclose(w[0, 1, :2], t(0.3302, 0.6698), atol=1e-4)
    assert torch.allclose(out[0, 1], t(0.3302, 0.6698), atol=1e-4)
    assert torch.allclose(w[0, 2], t(0.2483, 0.2483, 0.5035), atol=1e-4)
    assert torch.allclose(out[0, 2], t(0.7517, 0.7517), atol=1e-4)

    # exact: only the scaled scores are hand-derived
    s2 = t(0.5 ** 0.5, 0.5 ** 0.5, 2 ** 0.5)
    assert torch.allclose(w[0, 2], torch.softmax(s2, 0), atol=1e-12)

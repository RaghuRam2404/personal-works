import pytest
import torch
from gpt.attention import loop_attention, vector_attention

DT = torch.float64
ATOL = 1e-10

# (B, T, C, D, Dv, repeated_token)  -- x: [B, T, C], Wq/Wk: [D, C], Wv: [Dv, C]
CASES = [
    pytest.param((1, 1, 4, 4, 4, False), id="minimal"),
    pytest.param((3, 8, 8, 8, 8, False), id="batched"),
    pytest.param((2, 7, 6, 4, 5, False), id="D_ne_Dv"),
    pytest.param((2, 13, 8, 4, 4, False), id="longer_odd"),
    pytest.param((2, 6, 8, 4, 4, True), id="repeated_token"),
]
SEEDS = range(20)


@pytest.fixture(params=SEEDS, ids=lambda s: f"seed{s}")
def seed(request):
    return request.param


@pytest.fixture(params=CASES)
def case(request):
    return request.param


@pytest.fixture
def both(seed, case):
    """Run both implementations on identical inputs and return their intermediates."""
    B, T, C, D, Dv, repeated = case
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=g, dtype=DT)
    if repeated:
        x = x[:, :1, :].expand(B, T, C).clone()  # every row of x identical
    Wq = torch.randn(D, C, generator=g, dtype=DT)
    Wk = torch.randn(D, C, generator=g, dtype=DT)
    Wv = torch.randn(Dv, C, generator=g, dtype=DT)
    loop = loop_attention(x, Wq, Wk, Wv, return_intermediates=True)
    vec = vector_attention(x, Wq, Wk, Wv, return_intermediates=True)
    return case, loop, vec  # each: (Q, K, V, scores, weights, out)


def test_q_k_v_match(both):
    _, loop, vec = both
    for name, i in (("q", 0), ("k", 1), ("v", 2)):
        assert torch.allclose(loop[i], vec[i], atol=ATOL), name


def test_allowed_scores_match(both):
    (B, T, *_), loop, vec = both
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
    s_loop, s_vec = loop[3][:, allowed], vec[3][:, allowed]  # lower triangle only
    assert torch.isfinite(s_loop).all()  # loop scores are nan only above the diagonal
    assert torch.allclose(s_loop, s_vec, atol=ATOL)


def test_loop_scores_nan_above_diagonal(both):
    (B, T, *_), loop, _ = both
    above = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.isnan(loop[3][:, above]).all()


def test_weights_match(both):
    _, loop, vec = both
    assert torch.allclose(loop[4], vec[4], atol=ATOL)


def test_outputs_match(both):
    _, loop, vec = both
    assert torch.allclose(loop[5], vec[5], atol=ATOL)


def test_return_shapes_and_dtype(both):
    (B, T, C, D, Dv, _), loop, vec = both
    for r in (loop, vec):
        Q, K, V, scores, weights, out = r
        assert Q.shape == K.shape == (B, T, D)
        assert V.shape == out.shape == (B, T, Dv)
        assert scores.shape == weights.shape == (B, T, T)
        assert out.dtype == weights.dtype == DT


def test_default_return_is_out_and_weights(case, seed):
    B, T, C, D, Dv, _ = case
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=g, dtype=DT)
    Wq, Wk = (torch.randn(D, C, generator=g, dtype=DT) for _ in range(2))
    Wv = torch.randn(Dv, C, generator=g, dtype=DT)
    for fn in (loop_attention, vector_attention):
        out, weights = fn(x, Wq, Wk, Wv)
        full = fn(x, Wq, Wk, Wv, return_intermediates=True)
        assert torch.equal(out, full[5]) and torch.equal(weights, full[4])


def test_repeated_token_uniform_over_prefix(both):
    (B, T, C, D, Dv, repeated), loop, vec = both
    if not repeated:
        pytest.skip("only for the repeated-token case")
    for weights in (loop[4], vec[4]):
        for i in range(T):
            expected = torch.full((B, i + 1), 1 / (i + 1), dtype=DT)
            assert torch.allclose(weights[:, i, : i + 1], expected, atol=ATOL)


# ---------------------------------------------------------------------------
# Part 5 -- behavioral causal leakage test (both implementations)
# ---------------------------------------------------------------------------
IMPLS = [
    pytest.param(loop_attention, id="loop"),
    pytest.param(vector_attention, id="vector"),
]


def make_inputs(seed, B=2, T=8, C=6, D=4, Dv=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, C, generator=g, dtype=DT)
    Wq = torch.randn(D, C, generator=g, dtype=DT)
    Wk = torch.randn(D, C, generator=g, dtype=DT)
    Wv = torch.randn(Dv, C, generator=g, dtype=DT)
    return x, Wq, Wk, Wv, g


def replace_future(x, i, g):
    """Clone x and overwrite every position after i with unrelated values."""
    xnew = x.clone()
    xnew[:, i + 1:, :] = torch.randn(xnew[:, i + 1:, :].shape, generator=g, dtype=x.dtype) * 3 + 5
    return xnew


def past_is_unchanged(fn, x, Wq, Wk, Wv, i, g):
    """True if outputs and weights up to position i are unchanged by the future (nan counts as changed)."""
    out, w = fn(x, Wq, Wk, Wv)
    out2, w2 = fn(replace_future(x, i, g), Wq, Wk, Wv)
    return (torch.allclose(out[:, : i + 1], out2[:, : i + 1], atol=ATOL)
            and torch.allclose(w[:, : i + 1], w2[:, : i + 1], atol=ATOL))


@pytest.mark.parametrize("fn", IMPLS)
def test_future_does_not_change_past(fn, seed):
    x, Wq, Wk, Wv, g = make_inputs(seed)
    T = x.shape[1]
    i = int(torch.randint(0, T - 1, (1,), generator=g))  # random cut, at least one future position
    out, _ = fn(x, Wq, Wk, Wv)
    xnew = replace_future(x, i, g)
    out_new, _ = fn(xnew, Wq, Wk, Wv)
    assert torch.allclose(out[:, : i + 1], out_new[:, : i + 1], atol=ATOL)
    # the test must be able to see a difference: later outputs change
    assert not torch.allclose(out[:, i + 1:], out_new[:, i + 1:])


# ---------------------------------------------------------------------------
# Part 6 -- prove the tests have teeth (deliberately buggy vectorized variants)
# ---------------------------------------------------------------------------
def buggy_attention(x, Wq, Wk, Wv, bug):
    B, T, C = x.shape
    D = Wq.shape[0]
    q, k, v = x @ Wq.T, x @ Wk.T, x @ Wv.T
    if bug == "transpose_reshape":
        # k.reshape(B, D, T) always "fits" ([B,T,D] and [B,D,T] have the same number of elements),
        # but it re-chops memory instead of swapping axes (identity when T == D)
        scores = q @ k.reshape(B, D, T) / D ** 0.5
    else:
        scores = q @ k.transpose(-2, -1) / D ** 0.5
    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    if bug == "reversed_mask":
        scores = scores.masked_fill(causal_mask, float("-inf"))  # polarity reversed
    else:
        scores = scores.masked_fill(~causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-2 if bug == "softmax_dim_-2" else -1)
    return weights @ v, weights


def test_buggy_variant_without_bug_matches_vector_attention():
    x, Wq, Wk, Wv, _ = make_inputs(0)
    out, w = buggy_attention(x, Wq, Wk, Wv, bug=None)
    out_ref, w_ref = vector_attention(x, Wq, Wk, Wv)
    assert torch.allclose(out, out_ref, atol=ATOL) and torch.allclose(w, w_ref, atol=ATOL)


@pytest.mark.parametrize("seed", range(5))
def test_reversed_mask_is_caught_by_leakage_test(seed):
    x, Wq, Wk, Wv, g = make_inputs(seed)
    fn = lambda *a: buggy_attention(*a, bug="reversed_mask")
    assert not past_is_unchanged(fn, x, Wq, Wk, Wv, i=2, g=g)


@pytest.mark.parametrize("seed", range(5))
def test_softmax_wrong_axis_is_caught_by_row_sum_test(seed):
    x, Wq, Wk, Wv, _ = make_inputs(seed)
    _, w = buggy_attention(x, Wq, Wk, Wv, bug="softmax_dim_-2")
    B, T, _ = w.shape
    assert not torch.allclose(w.sum(-1), torch.ones(B, T, dtype=DT), atol=ATOL)


@pytest.mark.parametrize("seed", range(5))
def test_transpose_reshape_bug_when_T_equals_D(seed):
    # T == D == 4: the reshape is a no-op, so scores = q @ k (no transpose). Nothing raises a shape
    # error, and the result is still masked and normalized -- but it is wrong.
    x, Wq, Wk, Wv, g = make_inputs(seed, T=4, D=4)
    fn = lambda *a: buggy_attention(*a, bug="transpose_reshape")
    _, w = fn(x, Wq, Wk, Wv)
    B, T, _ = w.shape
    # NOT caught by the row-sum invariant (rows are still normalized) ...
    assert torch.allclose(w.sum(-1), torch.ones(B, T, dtype=DT), atol=ATOL)
    # ... but caught by parity, and by leakage: q @ k mixes positions, so k at future positions leaks in
    out_ref, w_ref = vector_attention(x, Wq, Wk, Wv)
    assert not torch.allclose(w, w_ref, atol=ATOL)
    assert not past_is_unchanged(fn, x, Wq, Wk, Wv, i=1, g=g)


@pytest.mark.parametrize("seed", range(5))
def test_transpose_reshape_bug_when_T_differs_from_D(seed):
    x, Wq, Wk, Wv, g = make_inputs(seed, T=8, D=4)
    fn = lambda *a: buggy_attention(*a, bug="transpose_reshape")
    _, w = fn(x, Wq, Wk, Wv)
    _, w_ref = vector_attention(x, Wq, Wk, Wv)
    assert not torch.allclose(w, w_ref, atol=ATOL)                   # parity catches it
    assert not past_is_unchanged(fn, x, Wq, Wk, Wv, i=2, g=g)         # distinct sizes also expose it to leakage

import torch

def loop_attention(
    x: torch.Tensor, #shape B, T, D
    Wq: torch.Tensor, #shape dq, D
    Wk: torch.Tensor, #shape dk, D and dk==dq
    Wv: torch.Tensor, #shape dv, D
    return_intermediates = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out [B, T, Dv], weights [B, T, T]).
    With return_intermediates: (Q, K, V, scores, weights, out); scores above the diagonal are nan."""


    if x.ndim == 2:
        x = x.unsqueeze(dim=0)
    
    B, T, D = x.shape
    dq = Wq.shape[0]
    dk = Wk.shape[0]
    dv = Wv.shape[0]
    
    Q = torch.zeros(size=(B,T,dq), dtype=x.dtype)
    for b in range(B):
        for i in range(T):
            for j in range(dq):
                for k in range(D):
                    Q[b,i,j] += x[b,i,k] * Wq.T[k,j]
    assert torch.allclose(Q, x@Wq.T)
    
    K = torch.zeros_like(Q)
    for b in range(B):
        for i in range(T):
            for j in range(dk):
                for k in range(D):
                    K[b,i,j] += x[b,i,k] * Wk.T[k,j]
    assert torch.allclose(K, x@Wk.T)
    
    V = torch.zeros(size=(B, T, dv), dtype=x.dtype)
    for b in range(B):
        for i in range(T):
            for j in range(dv):
                for k in range(D):
                    V[b,i,j] += x[b,i,k] * Wv.T[k,j]
    assert torch.allclose(V, x@Wv.T)

    """
    scores = torch.fill(torch.zeros(B, T, T), float("-inf"))
    for b in range(B):
        for i in range(T):
            for j in range(i+1):
                scores[b, i, j] = (Q[b, i] * K[b, j]).sum() / dk ** 0.5
    print(torch.softmax(scores, dim=2))
    """

    scores = torch.full((B, T, T), float("nan"), dtype=x.dtype)
    for b in range(B):
        for i in range(T):
            for j in range(i+1):
                scores[b, i, j] = (Q[b, i] * K[b, j]).sum() / dk ** 0.5
    #print(f"{scores=}")
    
    weights = torch.zeros_like(scores)
    for b in range(B):
        for i in range(T):
            row = scores[b, i, :i+1]
            e = torch.exp(row - row.max())
            e = e/e.sum()
            for j in range(e.shape[0]):
                weights[b,i,j] = e[j]
    #print(f"{weights=}") # shape (B, T, T)

    # Aggregation: Q and K only decided WHERE to look (the weights).
    # V is the content each token offers; weights[b, i, j] is how much of
    # token j's content token i takes. out[b, i] is a weighted average of the
    # value vectors of the tokens i can see (j <= i), so each row sums to 1
    # and the output stays on the scale of V.
    #
    # Example (T=3, dv=5), weights row 2 = [0.2, 0.3, 0.5]:
    #   out[2] = 0.2*V[0] + 0.3*V[1] + 0.5*V[2]
    #   out[2, 3] = 0.2*V[0,3] + 0.3*V[1,3] + 0.5*V[2,3]  (one output dim)
    # Token 0 sees only itself (weights [1, 0, 0]), so out[0] == V[0];
    # later tokens mix in more context. Zero weights above the diagonal mean
    # future tokens never leak into earlier outputs.
    
    out = torch.zeros(B, T, dv, dtype=x.dtype)

    for b in range(B):
        for i in range(T):
            acc = torch.zeros(dv, dtype=x.dtype)
            for j in range(i + 1):
                acc = acc + weights[b, i, j] * V[b, j]   # scalar * [dv] vector
            out[b, i] = acc

    assert torch.allclose(out, weights @ V)
    #print(out)  # shape (B, T, dv)

    if return_intermediates:
        return (Q, K, V, scores, weights, out)

    return (out, weights)


def vector_attention(
    x: torch.Tensor, #shape B, T, D
    Wq: torch.Tensor, #shape dq, D
    Wk: torch.Tensor, #shape dk, D and dk==dq
    Wv: torch.Tensor, #shape dv, D
    return_intermediates = False
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (out [B, T, Dv], weights [B, T, T])."""
    
    if x.ndim == 2:
        x = x.unsqueeze(dim=0)
    
    B, T, D = x.shape
    dq = Wq.shape[0]
    dk = Wk.shape[0]
    dv = Wv.shape[0]
    
    Q = x @ Wq.T # B, T, dq
    K = x @ Wk.T # B, T, dk
    V = x @ Wv.T # B, T, dv
    
    #find relevance
    scores = Q @ K.transpose(1,2) / dk ** 0.5
    causal_mask = torch.tril(torch.ones(T, T)) == 1.0
    masked_scores = scores.masked_fill(~ causal_mask, float("-inf"))
    weights = torch.softmax(masked_scores, dim=2)
    
    out = weights @ V
    
    if return_intermediates:
        return (Q, K, V, scores, weights, out)
    
    return (out, weights)

torch.manual_seed(10) 
B, T, D = 1, 3, 4
dq, dk, dv = 5, 5, 6
DT = torch.float64
x = torch.rand(size=(B, T, D), dtype=DT)
Wq = torch.rand(size=(dq, D), dtype=DT)
Wk = torch.rand(size=(dk, D), dtype=DT)
Wv = torch.rand(size=(dv, D), dtype=DT)
out1, weights1 = loop_attention(x, Wq, Wk, Wv)
print(f"{weights1=},\n{out1=}")
out2, weights2 = vector_attention(x, Wq, Wk, Wv)
print(f"{weights2=},\n{out2=}")
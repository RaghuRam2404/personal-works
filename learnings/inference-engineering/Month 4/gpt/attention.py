import torch
from torch import nn

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
    """
    x - Shape (B, T, D) - D considered as channels or embedding dim \n
    Wq - Shape (dq, D) \n
    Wk - Shape (dk, D) and dk == dq \n
    Wv - Shape (dv, D) \n
    return_intermediates - true to return Q, K, V \n
    Return (out [B, T, Dv], weights [B, T, T]). \n
    """
    
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
# %%
def multihead_loop(
    x: torch.Tensor, #shape B, T, C
    Wq: torch.Tensor, #shape dq, C
    Wk: torch.Tensor, #shape dk, C and dk==dq
    Wv: torch.Tensor, #shape dv, C
    n_head: int,
    return_intermediates = False,
    debug = False
):
    """
    x: [B, T, C]\n
    Wq, Wk : [Dq|Dk, C]\n
    Wv : [Dv, C]\n
    n_head: integer\n
    Returns (out [B, T, Dv], weights [B, H, T, T]).
    """
    
    if x.ndim == 2:
        x = x.unsqueeze(dim=0)
    
    B, T, C = x.shape
    dq = Wq.shape[0]
    dk = Wk.shape[0]
    dv = Wv.shape[0]
    assert dk == dq, f"{dq=} is not equal to {dk=}"
    
    D = C // n_head #not used, but just keeping it here
    DV = dv // n_head
    DK = dk // n_head
    DQ = dq // n_head
    
    assert C % n_head == 0 , f"{C=} is not a multiple of {n_head=}"
    assert dv % n_head == 0 , f"{dv=} is not a multiple of {n_head=}"
    assert dq % n_head == 0 , f"{dq=} is not a multiple of {n_head=}"
    assert dk % n_head == 0 , f"{dk=} is not a multiple of {n_head=}"

    out = torch.zeros((B, T, dv))
    weights = torch.zeros((B, n_head, T, T))
    if debug:
        print(f"in data shape is B={B},T={T},C={C}")
        print(f"weights shape should be {weights.shape}")
        print(f"out shape should be {out.shape}")
        print(f"{"="*20}")
    
    tweights, tout = [], []
    for h in range(n_head):
        if debug:
            print(f"Head {h+1}")
        Wqh = Wq[h*DQ : DQ*(h+1), :] # shape (Dq, C)
        Wkh = Wk[h*DK : DK*(h+1), :] # shape (Dk, C)
        Wvh = Wv[h*DV : DV*(h+1), :] # shape (DV, C)
        
        if debug:
            print(f"Wqh.shape:{Wqh.shape}")
            print(f"Wvh.shape:{Wvh.shape}")
        
        out_h, weights_h = vector_attention(x, Wqh, Wkh, Wvh)
        weights_h.unsqueeze_(dim=1)
        tweights.append(weights_h)
        tout.append(out_h)
        if debug:
            print(f"{weights_h.shape=},\n{out_h.shape=}")
            print(f"{"="*20}")
        
    weights = torch.cat(tweights, dim=1)
    out = torch.cat(tout, dim=2)
    
    if debug:
        print(f"{out.shape=},\n{weights.shape=}")
    
    return out, weights

"""
torch.manual_seed(10) 
B, T, C = 5, 3, 4 # C-channels instead of D-embedding dimension
#dq, dk, dv = 5, 5, 6
dq, dk, dv = 6, 6, 8
DT = torch.float64
x = torch.rand(size=(B, T, C), dtype=DT)
Wq = torch.rand(size=(dq, C), dtype=DT)
Wk = torch.rand(size=(dk, C), dtype=DT)
Wv = torch.rand(size=(dv, C), dtype=DT)

out, weights = multihead_loop(x, Wq, Wk, Wv, n_head=2, debug=True)
"""
# %%
import torch

def multihead_vectorized(
    x: torch.Tensor, #shape B, T, C
    Wq: torch.Tensor, #shape dq, C
    Wk: torch.Tensor, #shape dk, C and dk==dq
    Wv: torch.Tensor, #shape dv, C
    H: int,
    return_intermediates = False,
    debug = False
):
    """
    x: [B, T, C] \n
    Wq, Wk : [Dq|Dk, C] \n
    Wv : [Dv, C] \n
    H: integer \n
    Returns (out [B, T, Dv], weights [B, H, T, T]).
    """
    
    if x.ndim == 2:
        x = x.unsqueeze(dim=0)
    
    if debug:
        print(f"{x.shape=}, {Wq.shape=}")
    
    B, T, C = x.shape
    dv = Wv.shape[0]
    dk = Wk.shape[0]
    dq = Wq.shape[0]
    assert Wk.shape[1] == C, f"{Wk.shape[1]=} is not same as {C=}"
    assert Wq.shape[1] == C, f"{Wq.shape[1]=} is not same as {C=}"
    assert C%H == 0 and dv%H == 0 and dk%H == 0 and dq%H == 0 
    
    Dq = int(dq//H)
    Dk = int(dk//H)
    Dv = int(dv//H)
    
    Q = x @ Wq.T # B T dq
    Qh = Q.view(B, T, H, Dq).transpose(1,2) # B H T Dq
    K = x @ Wk.T # B T dk
    Kh = K.view(B, T, H, Dk).transpose(1,2) # B H T Dk
    V = x @ Wv.T # B T dv
    Vh = V.view(B, T, H, Dv).transpose(1,2) # B H T Dv
    
    if debug:
        print(f"{Q.shape=} and {Qh.shape=}")
        print(f"{V.shape=} and {Vh.shape=}")

    scores = Qh @ Kh.transpose(2,3) 
    scaled_scores = scores / Dk ** 0.5
    #tl = torch.tril(torch.ones_like(scaled_scores, dtype=torch.bool))
    tl = torch.tril(torch.ones(size=(T,T), dtype=torch.bool))
    masked_scores = torch.masked_fill(scaled_scores, ~tl, value=float("-inf"))
    weights = torch.softmax(masked_scores, dim=3)
    
    if debug:
        print(f"{weights.shape=} and {Vh.shape=}")

    pre_out = weights @ Vh
    out = pre_out.transpose(1,2).contiguous().view(B, T, dv)
    
    if debug:
        print(f"{pre_out.shape=}, {out.shape=}")
    
    if return_intermediates:
        return (Q, K, V, scores, scaled_scores, out, weights)
    
    return (out, weights)

# %%

class CausalSelfAttention(nn.Module):
    
    def __init__(self, n_embd, n_head, block_size, dropout, persist=False): 
        super().__init__()
        self.n_embd = n_embd # C
        self.n_head = n_head # H
        self.block_size = block_size # == T as of now
        self.dropout = dropout # dropout ratio
        
        assert n_embd % n_head == 0
        
        self.register_buffer("mask", 
                             torch.tril(torch.ones((block_size, block_size), dtype=torch.bool)),
                             persistent=persist)
        self.Wq = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float64)
        self.Wk = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float64)
        self.Wv = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float64)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False, dtype=torch.float64)
        
        self.D = n_embd // n_head # D = C // H
        
        self.attn_dropout = nn.Dropout(p = dropout)
        self.residual_dropout = nn.Dropout(p = dropout)
        
    def forward(self, x):
        if x.ndim == 2:
            x = x.unsqueeze(dim=0) 
        
        # here Dq, Dk, Dv are all same as of self.D
        
        B, T, C = x.shape
        assert T <= self.block_size, f"{T=} exceeds {self.block_size=}"
        H = self.n_head
        Dq, Dk, Dv = self.D, self.D, self.D
        
        Q = self.Wq(x)
        Qh = Q.view(B, T, H, Dq).transpose(1,2) # B H T Dq
        K = self.Wk(x)
        Kh = K.view(B, T, H, Dk).transpose(1,2) # B H T Dk
        V = self.Wv(x)
        Vh = V.view(B, T, H, Dv).transpose(1,2) # B H T Dv
        
        scores = Qh @ Kh.transpose(2,3) 
        scaled_scores = scores / Dk ** 0.5
        mask = ~self.get_buffer("mask")[:T, :T]
        masked_scores = torch.masked_fill(scaled_scores, mask, value=float("-inf"))
        weights = torch.softmax(masked_scores, dim=3)
        self.last_weights = weights
        attention_weights = self.attn_dropout(weights)
        
        pre_out = attention_weights @ Vh
        concat = pre_out.transpose(1,2).contiguous().view(B, T, C)
        self.last_y = concat
        y = self.c_proj(concat)
        last_y = self.residual_dropout(y)
        
        return last_y

torch.manual_seed(10) 
B, T, C = 2, 3, 4 # C-channels instead of D-embedding dimension
H = 2
csa = CausalSelfAttention(C,H,T,0.5)
x = torch.rand(size=(B, T, C), dtype=torch.float64)
out = csa(x)
print(out.shape)

# %%
if __name__ == "__main__":

    torch.manual_seed(10) 
    B, T, C = 2, 3, 4 # C-channels instead of D-embedding dimension
    #dq, dk, dv = 5, 5, 6
    dq, dk, dv = C+2, C+2, 8
    H = 2

    DT = torch.float64
    x = torch.rand(size=(B, T, C), dtype=DT)
    Wq = torch.rand(size=(dq, C), dtype=DT)
    Wk = torch.rand(size=(dk, C), dtype=DT)
    Wv = torch.rand(size=(dv, C), dtype=DT)

    #out, weights = multihead_vectorized(x, Wq, Wk, Wv, H, debug=True)


    Dq = int(dq/H)
    Dk = int(dk/H)
    Dv = int(dv/H)

    Q = x @ Wq.T # B T dq
    Qh = Q.view(B, T, H, Dq).transpose(1,2) # B H T dq//H
    K = x @ Wk.T # B T dk
    Kh = K.view(B, T, H, Dk).transpose(1,2) # B H T dk//H
    V = x @ Wv.T # B T dv
    Vh = V.view(B, T, H, Dv).transpose(1,2) # B H T dv//H

    scores = Qh @ Kh.transpose(2,3) 
    scaled_scores = scores / Dk ** 0.5
    tl = torch.tril(torch.ones_like(scaled_scores, dtype=torch.bool))
    masked_scores = torch.masked_fill(scaled_scores, ~tl, value=float("-inf"))
    weights = torch.softmax(masked_scores, dim=3)

    out = weights @ Vh
    print(out.shape, out.stride(), out.is_contiguous())
    # torch.Size([2, 2, 3, 4]) (24, 12, 4, 1) True

    out = out.transpose_(1,2)
    print(out.shape, out.stride(), out.is_contiguous())
    # torch.Size([2, 3, 2, 4]) (24, 4, 12, 1) False

    out = out.contiguous()
    print(out.shape, out.stride(), out.is_contiguous())
    # torch.Size([2, 3, 2, 4]) (24, 8, 4, 1) True

    out = out.transpose_(1,2).contiguous().view(B, T, dv)
    #out = out.transpose_(1,2).view(B, T, dv) #won't work

    out.shape
    #"""

    # %%
    out, weights = multihead_loop(x, Wq, Wk, Wv, n_head=2, debug=True)
    out1, weights1 = loop_attention(x, Wq, Wk, Wv)
    print(f"{weights1=},\n{out1=}")
    out2, weights2 = vector_attention(x, Wq, Wk, Wv)
    print(f"{weights2=},\n{out2=}")
# %%

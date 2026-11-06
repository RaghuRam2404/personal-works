import torch

vocab_size = 20

d = vocab_size
dq = 10
dk = 10

# %%
torch.manual_seed(10)
Qx = torch.randn(size=(dq, d))
Kx = torch.randn(size=(dk, d))
print(Qx.var(), Kx.var())

Q_K = Qx.T @ Kx # dxd output
print(Q_K.shape)
print(Q_K.var()) #same as of the vocab size
print(Q_K.sum(dim=1).var())
print((Q_K/d**0.5).sum(dim=1).var())

# %%
torch.manual_seed(10)
Q = torch.randn(10000,20)
K = torch.randn(10000,20)
print(Q.var(), K.var())
QK = Q * K
print(QK.var()) #variance is still 1
print(QK.sum(dim=1).var()) #but when we do rowwise addition, variance goes to 20
print((QK/20**0.5).sum(dim=1).var()) #variance back to 1

# %%

torch.manual_seed(10)
a = torch.randn(size=(1, 10))
print(a.var())
print(torch.softmax(a, dim=1))
print(torch.softmax(a*10, dim=1))
print(torch.softmax((a*10)/((a*10).var())**0.5, dim=1))

# %%
# Part 5: three-token hand example (causal, Wq = Wk = Wv = I, so q = k = v = x)
x = torch.tensor([[1., 0.],
                  [0., 1.],
                  [1., 1.]])
T, D = x.shape # (3,2), B=1 (ie) one batch
q = k = v = x # Wk, Wq, Wv are Identity matrix


for i in range(T):
    scores = [ q[i]*k[j] for j in range(i+1)] #dot product's multiplication with each of it's prev vectors
    stacked_scores = torch.stack(scores, dim=0) 
    print(f"{stacked_scores=}")
    stacked_scores_sum = stacked_scores.sum(dim=-1, keepdim=False) # dot product's sum
    print(f"{stacked_scores_sum=}")
    scaled_scores = stacked_scores_sum / (D ** 0.5)
    print(f"{scaled_scores=}")
    
    weights = torch.softmax(scaled_scores, dim=0)
    print(f"{weights=}")
    
    out = [weights[j] * v[j] for j in range(i+1)]
    print(f"{out=}")
    out = torch.stack(out, dim=0).sum(dim=0)
    print(f"{out=}")
    
    print()

# %%

from gpt.attention import loop_attention

torch.manual_seed(10) 
B, T, D = 1, 3, 4
dq, dk, dv = 5, 5, 6
x = torch.rand(size=(B, T, D))
Wq = torch.rand(size=(dq, D))
Wk = torch.rand(size=(dk, D))
Wv = torch.rand(size=(dv, D))
out, weights = loop_attention(x, Wq, Wk, Wv)

## Tests

assert weights.shape[1] == weights.shape[2]
B, T, T = weights.shape
dv = out.shape[2]
assert torch.allclose(weights.sum(dim=2, keepdim=True), torch.ones(size=(B,T,1)))
assert torch.allclose(torch.triu(weights, diagonal=1), torch.zeros_like(weights))

xnew = x.clone()
future_i = 2#changing the future tokens with random values
xnew[:,future_i:,:] = xnew[:,future_i:,:]+torch.randn((1,1,D))
out_new, weights_new = loop_attention(xnew, Wq, Wk, Wv)
assert torch.allclose(out[:,:future_i,:], out_new[:,:future_i,:])
assert not torch.allclose(out[:,future_i:,:], out_new[:,future_i:,:])
assert torch.allclose(weights[:,:future_i,:], weights_new[:,:future_i,:])
assert not torch.allclose(weights[:,future_i:,:], weights_new[:,future_i:,:])


Wq0 = torch.zeros(size=(dq, D))
out0, weights0 = loop_attention(x, Wq0, Wk, Wv)
for b in range(B):
    for i in range(T):
        prob_expected = float(1/(i+1))
        for j in range(i+1):
            assert weights0[b,i,j] == prob_expected
            

B, T, D = 1, 1, 4
dq, dk, dv = 5, 5, 6
x1 = torch.rand(size=(B, T, D))
Wq = torch.rand(size=(dq, D))
Wk = torch.rand(size=(dk, D))
Wv = torch.rand(size=(dv, D))
out1, weights1 = loop_attention(x1, Wq, Wk, Wv)
assert torch.allclose(out1, x1@Wv.T)

WqE = Wq * 1e4
outE, weightsE = loop_attention(x, WqE, Wk, Wv)
assert torch.isfinite(outE).all()


# %%

a = torch.tensor([[1,2,3]])
a.T[1,0]

b = torch.full(size=(2,3), fill_value=float("-inf"))
b[0,0] = 10
b[1,0] = 5
b[1,1] = 9
print(b)
print(torch.softmax(b, dim=1))

c = torch.randn(size=(1,2,3))
c.view(1,-1).shape

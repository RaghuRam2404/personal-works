import torch
from torch import nn
from gpt.attention import multihead_vectorized

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

out, weights = multihead_vectorized(x, Wq, Wk, Wv, H)

linear = nn.Linear(5, 10)
print(linear.weight.shape)
a = torch.randn((3,5))
linear(a).shape

a = torch.randint(0, 100, size=(20,), dtype=torch.float)
print(a, a.sum())
drop = nn.Dropout(p=0.5)
drop.train()
ad = drop(a)
print(ad, ad.sum())
drop.eval()
ad = drop(a)
print(ad, ad.sum())

print(a.sum())
for r in [10, 100, 1000, 10000]:
    sums = [drop(a).sum().item() for _ in range(r)]
    print(sum(sums) / len(sums))  # similar to a.sum()
    

class M(nn.Module):
    def __init__(self, block_size, persist=False):
        super().__init__()
        self.register_buffer("mask", 
                             torch.tril(torch.ones(size=(block_size, block_size), dtype=torch.bool)),
                             persistent=persist)
        
m = M(4, persist=False)
m.state_dict()

############

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

out, weights = multihead_vectorized(x, Wq, Wk, Wv, H)
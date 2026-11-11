import torch

B, T, C = 2, 3, 4
a = torch.randn((B, T, C))

a_new = torch.zeros_like(a)
for b in range(B):
    for t in range(T):
        xprev = a[b, :t+1]
        xprev = xprev.mean(dim=0)
        a_new[b,t] = xprev

a, "\n", a_new

mask = torch.triu(torch.ones(size=(T,T)), diagonal=0)
mask = mask / mask.sum(0)
a_new1 = (a.transpose(1,2) @ mask).transpose(1,2)

mask = torch.tril(torch.ones(size=(T,T)), diagonal=0)
mask = mask / mask.sum(1, keepdim=True)
a_new2 = mask @ a
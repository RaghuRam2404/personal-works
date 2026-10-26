import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

############ DATALOAD AN PREREQUISITE

MANUAL_SEED = 42

def get_words(filename = "../Month 3/names.txt"):
    words = []
    with open(filename, 'r') as file:
        for line in file:
            words.append(line.strip())
    return words

words = get_words()
chars = set()
pairs = {}
eow = "."

for word in words:
    word = [eow]+list(word)+[eow]
    for c1, c2 in zip(list(word), list(word[1:])):
        chars.add(c1); chars.add(c2)
        pair = (c1, c2)
        pairs[pair] = pairs.get(pair, 0) + 1

chars = [eow] + list(sorted(chars)[1:])
itos, stoi = {}, {}
for idx, c in enumerate(chars):
    itos[idx] = c
    stoi[c] = idx

############ CREATING DATASETS
    
nchars = len(chars)
block_size = 3

xs, ys = [], []
for word in words:
    word = [eow]+list(word)+[eow]
    for i in range(len(word)-block_size):
        tx = word[i:i+block_size]
        tx = [stoi[s] for s in tx]
        ty = stoi[word[i+block_size]]
        xs.append(tx)
        ys.append(ty)
xs, ys = torch.tensor(xs), torch.tensor(ys)
xs.shape, ys.shape

n = xs.shape[0]
n1 = int(n*0.8)
n2 = n1 + int(n*0.1)

xtrain, ytrain = xs[0:n1], ys[0:n1]
xdev, ydev = xs[n1:n2], ys[n1:n2]
xtest, ytest = xs[n2:n], ys[n2:n]

############ EMBEDDING TEST

emb_size = 10
C = torch.randn(size=(nchars, emb_size))
C[[0,0,1]].view(-1).shape

############ MODEL WITH BATCH NORM

def compare(input, l1, l2):
    assert l1.shape == l2.shape, f"both grads have diff shape {l1.shape=} and {l2.shape=}"
    exact = torch.all(l1 == l2).item()
    approximate = torch.allclose(l1, l2)
    max_diff = (l1 - l2).abs().max().item()
    print(f"{input:20s} | exact: {str(exact):5s} | approximate: {str(approximate):5s} | max_diff: {max_diff:.7f}")

t1 = torch.randn(size=(10,10))
t2 = torch.randn(size=(10,10))
#compare("t1 and t2", t1, t1.view(-1))

# %%
"""
############ BREAKING DOWN THE CROSS ENTROPY LOSS
g = torch.Generator().manual_seed(MANUAL_SEED)

logits = torch.randn(size=(100, 20), generator=g)
targets = torch.randint(0,20, size=(100,),generator=g)
fast_loss = torch.nn.CrossEntropyLoss()(logits, targets)
print(f"fast loss: {fast_loss:.7f}")

logits_reduced = logits - logits.max(dim=1, keepdim=True).values
logits_exp = logits_reduced.exp()
logits_exp_sum = logits_exp.sum(dim=1, keepdim=True)
probs = logits_exp/logits_exp_sum
onehot = probs[torch.arange(0, probs.shape[0]), targets]
onehot_log = onehot.log()
nll = -onehot_log.mean()
print(f"nll loss: {nll:.7f}")
"""
# %%
############ FORWARD PASS MODEL WITH BATCH NORM

n_hidden = 64
batch_size = 32
emb_size = 10
momentum = 0.001
epsilon = 1e-5
loss_fn = torch.nn.CrossEntropyLoss()

g = torch.Generator().manual_seed(MANUAL_SEED)

C = torch.randn(size=(nchars, emb_size), generator=g)
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g)
W1 = W1 * (5/3)/((emb_size * block_size) ** 0.5)

b1 = torch.randn(size=(1,n_hidden), generator=g) * 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0.01
b2 = torch.randn(size=(1,nchars), generator=g) * 0.01

running_mean = torch.zeros(size=(1, n_hidden))
running_std = torch.ones(size=(1,n_hidden))
bngain = torch.ones(size=(1,n_hidden))
bnbias = torch.zeros(size=(1,n_hidden))

params = [C, W1, b1, W2, b2, bngain, bnbias]
for param in params:
    param.requires_grad_(True)

#zero grad
for param in params:
    param.grad = None

idx = torch.randint(low=0, high=xtrain.shape[0], size=(batch_size,), generator=g)
xdata = xtrain[idx]
ydata = ytrain[idx]

#forward pass
embed = C[xdata] #done
embedcat = embed.view(-1, block_size*emb_size) #done
l1 = embedcat @ W1 + b1 #done
batch_mean = l1.mean(dim=0, keepdim=True) #done
batch_diff = l1 - batch_mean #done
batch_diff_sq = batch_diff ** 2 #done
batch_var = 1/(n-1) * batch_diff_sq.sum(dim=0, keepdim=True) #done
batch_std_inv = (batch_var + epsilon) ** -0.5 #done

centered_l1 = l1 - batch_mean #done
l1_norm = centered_l1 * batch_std_inv #done
h1_preact = bngain * l1_norm + bnbias #done
h1 = h1_preact.tanh() #done
logits = h1 @ W2 + b2 #done
logits_max = logits.max(dim=1, keepdim=True).values #done
logits_reduced = logits - logits_max #done
logits_exp = logits_reduced.exp() #done
logits_exp_sum = logits_exp.sum(dim=1, keepdim=True) #done
logits_exp_sum_inv = logits_exp_sum**-1 #done
probs = logits_exp * logits_exp_sum_inv #done
onehot_probs = probs[torch.arange(0, probs.shape[0]), ydata] #done
log_probs = onehot_probs.log() #done
loss = -log_probs.mean() #done
print(f"loss: {loss.item():.7f}")
#print(f"fast loss: {loss_fn(logits, ydata).item():.7f}")

retain = [embed, embedcat, l1, W1, b1, batch_mean,batch_diff, batch_diff_sq, batch_var, batch_std_inv, l1, centered_l1, l1_norm, h1_preact, h1, logits, logits_max, 
          logits_reduced, logits_exp, logits_exp_sum, W2, b2,
          logits_exp_sum_inv, probs, onehot_probs, log_probs, loss]
for r in retain:
    r.retain_grad()

loss.backward()
# %%
############ MANUAL BACKPROP

d_log_probs = -torch.ones_like(log_probs)/len(log_probs)
d_onehot_probs = onehot_probs ** -1 * d_log_probs
d_probs = torch.zeros_like(probs)
d_probs[torch.arange(0, probs.shape[0]), ydata] = d_onehot_probs
d_logits_exp_sum_inv = (logits_exp * d_probs).sum(dim=1, keepdim=True)
d_logits_exp = d_probs * logits_exp_sum_inv
d_logits_exp_sum = -1*logits_exp_sum**-2 * d_logits_exp_sum_inv
d_logits_exp += torch.ones_like(logits_exp_sum) * d_logits_exp_sum
d_logits_reduced = logits_exp * d_logits_exp
d_logits = d_logits_reduced.clone()
d_logits_max = -d_logits_reduced.sum(dim=1, keepdim=True)
d_logits += (logits == logits_max).float() * d_logits_max
d_b2 = d_logits.sum(dim=0, keepdim=True)
d_W2 = h1.T @ d_logits
d_h1 = d_logits @ W2.T
d_h1_preact = (1-h1**2) * d_h1
d_bnbias = d_h1_preact.sum(dim=0, keepdim=True)
d_bngain = (l1_norm * d_h1_preact).sum(dim=0, keepdim=True)
d_l1_norm = d_h1_preact * bngain
d_batch_std_inv = (centered_l1 * d_l1_norm).sum(dim=0, keepdim=True)
d_centered_l1 = d_l1_norm * batch_std_inv
d_batch_mean = -1 * d_centered_l1.sum(dim=0, keepdim=True)
d_l1 = d_centered_l1.clone()
d_batch_var = -0.5 * (batch_var + epsilon) ** -1.5 * d_batch_std_inv
d_batch_diff_sq = (1/(n-1)) * torch.ones_like(batch_diff_sq) * d_batch_var
d_batch_diff = 2 * batch_diff * d_batch_diff_sq
d_batch_mean += -1 * d_batch_diff.sum(dim=0, keepdim=True)
d_l1 += d_batch_diff.clone()
d_l1 += torch.ones_like(l1)/l1.shape[0] * d_batch_mean
d_b1 = d_l1.sum(dim=0, keepdim=True)
d_W1 = embedcat.T @ d_l1
d_embedcat = d_l1 @ W1.T
d_embed = d_embedcat.view(-1, block_size, emb_size)
d_C = torch.zeros_like(C)
for i in range(xdata.shape[0]):
    for j in range(xdata.shape[1]):
        d_C[xdata[i,j]] += d_embed[i,j]

C.shape, embed.shape, xdata.shape

compare("d_log_probs", d_log_probs, log_probs.grad)
compare("d_onehot_probs", d_onehot_probs, onehot_probs.grad)
compare("d_probs", d_probs, probs.grad)
compare("d_logits_exp_sum_inv", d_logits_exp_sum_inv, logits_exp_sum_inv.grad)
compare("d_logits_exp_sum", d_logits_exp_sum, logits_exp_sum.grad)
compare("d_logits_exp", d_logits_exp, logits_exp.grad)
compare("d_logits_reduced", d_logits_reduced, logits_reduced.grad)
compare("d_logits", d_logits, logits.grad)
compare("d_logits_max", d_logits_max, logits_max.grad)
compare("d_b2", d_b2, b2.grad)
compare("d_W2", d_W2, W2.grad)
compare("d_h1", d_h1, h1.grad)
compare("d_h1_preact", d_h1_preact, h1_preact.grad)
compare("d_bnbias", d_bnbias, bnbias.grad)
compare("d_bngain", d_bngain, bngain.grad)
compare("d_l1_norm", d_l1_norm, l1_norm.grad)
compare("d_batch_std_inv", d_batch_std_inv, batch_std_inv.grad)
compare("d_centered_l1", d_centered_l1, centered_l1.grad)
compare("d_batch_std_inv", d_batch_std_inv, batch_std_inv.grad)
compare("d_batch_diff_sq", d_batch_diff_sq, batch_diff_sq.grad)
compare("d_batch_diff", d_batch_diff, batch_diff.grad)
compare("d_batch_mean", d_batch_mean, batch_mean.grad)
compare("d_l1", d_l1, l1.grad)
compare("d_b1", d_b1, b1.grad)
compare("d_W1", d_W1, W1.grad)
compare("d_embedcat", d_embedcat, embedcat.grad)
compare("d_embed", d_embed, embed.grad)
compare("d_C", d_C, C.grad)
# %%

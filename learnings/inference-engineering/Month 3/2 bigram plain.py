import os
import torch
import numpy

MANUAL_SEED = 2147483647

def get_words(filename = "./names.txt"):
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

#sorted(pairs.items(), key=lambda kv: -kv[1])

chars = [eow] + list(sorted(chars)[1:])
itos, stoi = {}, {}
for idx, c in enumerate(chars):
    itos[idx] = c
    stoi[c] = idx
#print(len(chars), stoi, itos)

bigram_matrix = torch.zeros(size=(len(chars), len(chars)), dtype=torch.float32)
for (c1, c2), value in pairs.items():
    bigram_matrix[stoi[c1], stoi[c2]] = value

#Normalize, so each row is a probablity row of what happens after one alphabet
N = bigram_matrix+1.0
N = N/N.sum(dim=1, keepdim=True)

g = torch.Generator().manual_seed(MANUAL_SEED)
for i in range(5):
    out = []
    idx = 0
    while True:    
        idx = torch.multinomial(N[idx], num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print("".join(out))
    

"""
cexze
momasurailezityha
konimittain
llayn
ka
"""
    

#Loss
log_likelihood = 0.0
count = 0 

for word in words:
    word = [eow]+list(word)+[eow]
    for c1, c2 in zip(list(word), list(word[1:])):
        prob = torch.log(N[stoi[c1], stoi[c2]])
        log_likelihood += prob
        count += 1
nll = -log_likelihood/count
print(f"nll loss for count based model is {nll}")

###########################################

### Plain neuron
xs = []
ys = []

for word in words:
    word = [eow] + list(word) + [eow]
    for c1, c2 in zip(word, word[1:]):
        xs.append(stoi[c1])
        ys.append(stoi[c2])
        
print(f"{len(xs)=}, {len(ys)=}")
xs = torch.tensor(xs)
ys = torch.tensor(ys)

g = torch.Generator().manual_seed(MANUAL_SEED)
w = torch.randn((len(chars), len(chars)), requires_grad=True, generator=g) #prob matrix

# 1 round of training
epochs = 100 #running for 100 epochs will lead to model learning the distribution of the original input
for i in range(epochs):
    xenc = torch.nn.functional.one_hot(xs, num_classes=len(chars)).float()
    logits = (xenc @ w) #similar to the count hash map we had earlier
    counts = logits.exp() # similar to the count
    prob = counts / counts.sum(dim=1, keepdim=True) # each combination has 27prob of what's the next character
    nll = - prob[torch.arange(xenc.shape[0]), ys].log().mean() + 0.01*(w**2).mean()
    print(nll) #It's decreasing really
    #print(xenc.shape, prob.shape, xenc*prob)
    
    w.grad = None
    nll.backward()
    
    w.data += -50*w.grad
    
print(f"nll loss for nn based model is {nll}")
    
g = torch.Generator().manual_seed(MANUAL_SEED)
# prediction
for i in range(5):
    out = []
    idx = 0
    
    while True:
        xt = torch.nn.functional.one_hot(torch.tensor([idx]), num_classes=len(chars)).float()
        logit = (xt @ w).exp()
        probs = logit/logit.sum(dim=1, keepdim=True) #it tells probs of choose next character
        
        idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
        if idx == 0:
            break
        out.append(itos[idx])
    print("".join(out))
    
""" output when run for epochs 100
cexze
momasurailezityha
konimittain
llayn
ka
"""

P = torch.softmax(w.detach(), dim=1)                    # neural probabilities, rows sum to 1
print((P - N).abs().max(), (P - N).abs().mean())        # table-to-table gap
print(-P[xs, ys].log().mean())                          # pure NLL without the L2 term
# sanity: seeding W with log-probabilities should reproduce N exactly
print((torch.softmax(N.log(), dim=1) - N).abs().max())  # ~1e-8

import torch
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
import matplotlib.pyplot as plt

############ DATALOAD AN PREREQUISITE

MANUAL_SEED = 42

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

############ TESTING SAMPLE MODEL AND LOSS FUNCTION

g = torch.Generator().manual_seed(MANUAL_SEED)
W1 = torch.randn(size=(emb_size * block_size, 200), generator=g)
b1 = torch.randn(size=(200,), generator=g)
W2 = torch.randn(size=(200,nchars), generator=g)
b2 = torch.randn(size=(nchars,), generator=g)

tdata = C[xs[:5]].view(-1, emb_size*block_size)
samplelogits = (((tdata @ W1 + b1)@W2)+b2)
loss_fn = CrossEntropyLoss()
loss_fn(samplelogits, ys[:5])

############ BUILDING MODEL & TRAINING IT

g = torch.Generator().manual_seed(MANUAL_SEED)

C = torch.randn(size=(nchars, emb_size), requires_grad=True)
W1 = torch.randn(size=(emb_size * block_size, 200), requires_grad=True, generator=g)
b1 = torch.randn(size=(200,), requires_grad=True, generator=g)
W2 = torch.randn(size=(200,nchars), requires_grad=True, generator=g)
b2 = torch.randn(size=(nchars,), requires_grad=True, generator=g)

params = [C, W1, b1, W2, b2]

epochs = 200000
batch_size = 8
loss_fn = CrossEntropyLoss()
losses = []

for epoch in range(epochs):
    
    for param in params:
        param.grad = None
    
    idx = torch.randint(low=0, high=xtrain.shape[0], size=(batch_size,))
    xdata = xtrain[idx]
    ydata = ytrain[idx]

    embed = C[xdata].view(-1, block_size*emb_size)
    l1 = embed @ W1 + b1
    h1 = l1.tanh()
    logits = h1 @ W2 + b2
    
    loss = loss_fn(logits, ydata)
    losses.append(loss.item())
    
    loss.backward()
    lr = 0.1 if epoch < 100000 else 0.01
    
    for param in params:
        param.data += -lr * param.grad

#Losses gruped by {group_count}
group_count = 1000
avg_losses = torch.tensor(losses).view(-1,group_count).sum(dim=1, keepdim=True)/group_count
plt.plot(avg_losses)
plt.show()

print(avg_losses[-1]) #2.0340

############ VISUALIZING EMBEDDINGS

plt.title("Embedding area")
plt.scatter(x = C[:,0].data, y = C[:,1].data, s=200)
for i in range(C.shape[0]):
    plt.text(x = C[i,0].data, y = C[i,1].data, s=itos[i], ha="center", va="center", color='white')
plt.grid('minor')
plt.show()

############ TRAIN DEV TEST LOSS

def get_loss(data, target):
    with torch.no_grad():
        embed = C[data].view(-1, block_size*emb_size)
        l1 = embed @ W1 + b1
        h1 = l1.tanh()
        logits = h1 @ W2 + b2
        
        loss = loss_fn(logits, target)
    return loss

train_loss = get_loss(xtrain, ytrain)
dev_loss = get_loss(xdev, ydev)
test_loss = get_loss(xtest, ytest)
print(f"{train_loss=}, {dev_loss=} and {test_loss=}") 
#train_loss=tensor(2.0133), dev_loss=tensor(2.4013) and test_loss=tensor(2.4281)

############ GENERATING THE NAMES

out_n = 10
g = torch.Generator().manual_seed(MANUAL_SEED)

for i in range(out_n):
    with torch.no_grad():
        out = []
        blocks = [0] * block_size
        
        while True:

            embed = C[blocks[-3:]].view(-1, block_size * emb_size)
            logits = ((embed @ W1 + b1).tanh()) @ W2 + b2
            probs = F.softmax(logits, dim=1)
            idx = torch.multinomial(probs, num_samples=1, replacement=True, generator=g).item()
            if idx == 0:
                break
            out.append(itos[idx])
            blocks.append(idx)

        print("".join(out))
        
"""
Output generated:
 
iyuselyn
ismar
idredyn
sin
milla
mila
joshwan
ina
iyah
oda
"""
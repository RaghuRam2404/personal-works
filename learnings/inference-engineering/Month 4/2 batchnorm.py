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

g = torch.Generator().manual_seed(MANUAL_SEED)
n_hidden = 200

C = torch.randn(size=(nchars, emb_size), requires_grad=True)
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g)
W1 = W1 * (5/3)/((emb_size * block_size) ** 0.5)

b1 = torch.randn(size=(n_hidden,), generator=g) * 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0.01

running_mean = torch.zeros(size=(1, n_hidden))
running_std = torch.ones(size=(1,n_hidden))
bngain = torch.ones(size=(1,n_hidden), requires_grad=True)
bnbias = torch.zeros(size=(1,n_hidden), requires_grad=True)
momentum = 0.001
epsilon = 1e-5
params = [C, W1, b1, W2, b2, bngain, bnbias]

# Enable gradients after scaling so every parameter remains a leaf tensor.
for param in params:
    param.requires_grad_(True)

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
    l1.shape
    #### BATCH NORM LAYER
    batch_mean = l1.mean(dim=0, keepdim=True)
    batch_std = l1.std(dim=0, keepdim=True)
    l1 = (l1 - batch_mean)/(batch_std + epsilon)
    h1_preact = bngain * l1 + bnbias
    
    with torch.no_grad():
        running_mean = (1-momentum) * running_mean + momentum * batch_mean
        running_std = (1-momentum) * running_std + momentum * batch_std
    #### END OF BATCH NORM
    
    h1 = h1_preact.tanh()
    logits = h1 @ W2 + b2
    
    loss = loss_fn(logits, ydata)
    loss.backward()
    
    losses.append(loss.item())
    lr = 0.1 if epoch < 100000 else 0.01
    
    with torch.no_grad():
        for param in params:
            param += -lr * param.grad
        

#Losses gruped by {group_count}
group_count = 1000
avg_losses = torch.tensor(losses).view(-1,group_count).sum(dim=1, keepdim=True)/group_count
plt.plot(avg_losses)
plt.show()

print(avg_losses[-1]) #2.1733

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
        l1 = l1.tanh()
        h1_preact = bngain * (l1 - running_mean)/(running_std+epsilon) + bnbias
        logits = h1_preact @ W2 + b2
        
        loss = loss_fn(logits, target)
    return loss

train_loss = get_loss(xtrain, ytrain)
dev_loss = get_loss(xdev, ydev)
test_loss = get_loss(xtest, ytest)
print(f"{train_loss=}, {dev_loss=} and {test_loss=}") 
#train_loss=tensor(3.6285), dev_loss=tensor(4.0111) and test_loss=tensor(4.0076)
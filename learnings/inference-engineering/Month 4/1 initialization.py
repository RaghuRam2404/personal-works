# %%
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt

# %%
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

# %%
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

# %%
############ MODEL

g = torch.Generator().manual_seed(MANUAL_SEED)
emb_size = 10
n_hidden = 200

C = torch.randn(size=(nchars, emb_size), requires_grad=True, generator=g)
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g) * 0 #* 0.01
W1 = W1 * (5/3)/((emb_size * block_size) ** 0.5)

b1 = torch.randn(size=(n_hidden,), generator=g) * 0 #* 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0 #* 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0

params = [C, W1, b1, W2, b2]
for param in params:
    param.requires_grad_(True)

epochs = 200000
batch_size = 32
loss_fn = CrossEntropyLoss()
losses = []

for epoch in range(epochs):
    
    for param in params:
        param.grad = None
    
    idx = torch.randint(low=0, high=xtrain.shape[0], size=(batch_size,)) #, generator=g)
    xdata = xtrain[idx]
    ydata = ytrain[idx]

    embed = C[xdata].view(-1, block_size*emb_size)
    l1 = embed @ W1 + b1
    
    h1 = l1.tanh()
    logits = h1 @ W2 + b2
    
    loss = loss_fn(logits, ydata)
    loss.backward()
    
    if epoch%10000 == 0:
        print(f"Epoch: {epoch:5d}/{epochs:5d} - Loss: {loss.item():.4f}")
    
    losses.append(loss.item())
    lr = 0.1 if epoch < 100000 else 0.01
    
    with torch.no_grad():
        for param in params:
            param += -lr * param.grad
        
    break

# %%

values_above_0_97 = (h1.view(-1).abs() > 0.97).float().sum()/len(h1.view(-1).tolist()) * 100
print(f"{values_above_0_97=}")

plt.figure(figsize=(10,6))
plt.imshow(h1.abs() > 0.97, cmap='gray', interpolation='nearest') #white - true, black - false
plt.show()

gain = 5/3 #fight the squeezing of the non-linearity function with gain
x = torch.randn(100,200)
w = torch.randn(200,100) * gain / 200**0.5
xw = x@w
h = xw.tanh()

print(f"mean(x):{x.view(-1).mean().item()} and std(x): {x.view(-1).std().item()}")
print(f"mean(xw):{xw.view(-1).mean().item()} and std(xw): {xw.view(-1).std().item()}")
print(f"mean(h):{h.view(-1).mean().item()} and std(h): {h.view(-1).std().item()}")

plt.subplot(1,2,1)
plt.hist(x.view(-1).tolist(), 50)
plt.subplot(1,2,2)
plt.hist(xw.view(-1).tolist(), 50)
plt.show()

# %%
#Losses gruped by {group_count}
group_count = 1000
avg_losses = torch.tensor(losses).view(-1,group_count).sum(dim=1, keepdim=True)/group_count
plt.plot(avg_losses)
plt.show()

print(avg_losses[-1])

# %%
############ TRAIN DEV TEST LOSS

def get_loss(data, target):
    with torch.no_grad():
        embed = C[data].view(-1, block_size*emb_size)
        l1 = embed @ W1 + b1
        l1 = l1.tanh()

        logits = l1 @ W2 + b2
        
        loss = loss_fn(logits, target)
    return loss

train_loss = get_loss(xtrain, ytrain)
dev_loss = get_loss(xdev, ydev)
test_loss = get_loss(xtest, ytest)
print(f"{train_loss=}, {dev_loss=} and {test_loss=}") 

# %%
############ LOSS LOGS

"""
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g)
b1 = torch.randn(size=(n_hidden,), generator=g)
W2 = torch.randn(size=(n_hidden,nchars), generator=g)
b2 = torch.randn(size=(nchars,), generator=g)
"""
# with no initialization optimization, all the weights are initialized as randn
# train_loss=tensor(1.9043), dev_loss=tensor(2.3265) and test_loss=tensor(2.3657)
# first epoch mini-batch loss: 23.5189

"""
print(-torch.log(torch.tensor(1/27)))
"""
# initial loss should be around 3.2958 

"""
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g)
b1 = torch.randn(size=(n_hidden,), generator=g)
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0
"""
# by removing the confidence of the last layer to not give wide range of output
# which lead to higher errors
# higher values in the logits, leading to erratic bigger losses
# first epoch mini-batch loss: 3.3221
# train_loss=tensor(1.8449), dev_loss=tensor(2.3123) and test_loss=tensor(2.3599)

"""
values_above_0_97 = (h1.view(-1).abs() > 0.97).float().sum()/len(h1.view(-1).tolist()) * 100
print(f"{values_above_0_97=}")

plt.figure(figsize=(10,6))
plt.imshow(h1.abs() > 0.97, cmap='gray', interpolation='nearest') #white - true, black - false
plt.show()
"""
# values_above_0_97=tensor(68.4219) #after first epoch
# these much neurons are saturated, meaning that the learning is not going through
# after the backprop

"""
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g) * 0.01
b1 = torch.randn(size=(n_hidden,), generator=g) * 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0
"""
# values_above_0_97=tensor(0.) #after first epoch
# the entire image is now black
# After full training, losses are
# train_loss=tensor(1.8866), dev_loss=tensor(2.2932) and test_loss=tensor(2.3144)


"""
x = torch.randn(100,200)
w = torch.randn(200,100) * 0.2
xw = x@w

print(f"mean(x):{x.view(-1).mean().item()} and std(x): {x.view(-1).std().item()}")
print(f"mean(xw):{xw.view(-1).mean().item()} and std(xw): {xw.view(-1).std().item()}")

plt.subplot(1,2,1)
plt.hist(x.view(-1).tolist(), 50)
plt.subplot(1,2,2)
plt.hist(xw.view(-1).tolist(), 50)
plt.show()
"""
# without *0.2
# mean(x):-0.00047394257853738964 and std(x): 1.0078907012939453
# mean(xw):0.08676149696111679 and std(xw): 14.308067321777344
# with *0.2, the std got reduced (but not exactly to 1), but what's the exact number to mulitply with?
# mean(x):0.0037310051266103983 and std(x): 0.9890946745872498
# mean(xw):-0.008863536641001701 and std(xw): 2.8131487369537354

"""
x = torch.randn(100,200)
w = torch.randn(200,100) / 200**0.5 #sqrt of fan-in neurons count
"""
# mean(x):0.0063748667016625404 and std(x): 0.9982388615608215
# mean(xw):-0.004570898599922657 and std(xw): 1.0134479999542236

"""
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g)
W1 = W1 * (5/3)/((emb_size * block_size) ** 0.5) #scale the std of W1

b1 = torch.randn(size=(n_hidden,), generator=g) * 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0
"""
# Fight the non-linearity function's squeezing with the gain (ie) scale the std up
# dividing by the sqrt of fan-in
# train_loss=tensor(1.8089), dev_loss=tensor(2.2801) and test_loss=tensor(2.3234)


"""
W1 = torch.randn(size=(emb_size * block_size, n_hidden), generator=g) * 0 #* 0.01
b1 = torch.randn(size=(n_hidden,), generator=g) * 0 #* 0.01
W2 = torch.randn(size=(n_hidden,nchars), generator=g) * 0 #* 0.01
b2 = torch.randn(size=(nchars,), generator=g) * 0
"""
# all weights zero init
# train_loss=tensor(2.6580), dev_loss=tensor(2.8156) and test_loss=tensor(2.8110)

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
block_size = 8

xs, ys = [], []
for idx, word in enumerate(words):
    word = ([eow]*block_size)+list(word)+[eow]
    for i in range(len(word)-block_size):
        tx = word[i:i+block_size]
        tx = [stoi[s] for s in tx]
        ty = stoi[word[i+block_size]]
        xs.append(tx)
        ys.append(ty)
    
#print(xs)
#
# print(ys)

xs, ys = torch.tensor(xs), torch.tensor(ys)
xs.shape, ys.shape

n = xs.shape[0]
n1 = int(n*0.8)
n2 = n1 + int(n*0.1)

xtrain, ytrain = xs[0:n1], ys[0:n1]
xdev, ydev = xs[n1:n2], ys[n1:n2]
xtest, ytest = xs[n2:n], ys[n2:n]

# %%
############# LAYERS


class Linear:
    def __init__(self, in_features, out_features, bias=True, generator=None):
        self.fan_in = in_features
        self.w = torch.randn(size=(in_features, out_features), generator=generator) / in_features ** 0.5
        self.b = None
        if bias:
            self.b = torch.randn(size=(1, out_features), generator=generator)
    
    def __call__(self, x):
        self.out = x @ self.w
        if self.b is not None:
            self.out += self.b
        return self.out
    
    def parameters(self):
        return [self.w] + ([] if self.b is None else [self.b])
    
class BatchNorm1D:
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        self.training = True
        self.num_features = num_features
        self.epsilon = eps
        self.momentum = momentum

        self.bngain = torch.ones(size=(1, num_features))
        self.bnbias = torch.zeros(size=(1, num_features))

        self.running_mean = torch.zeros(size=(1, num_features))
        self.running_var = torch.ones(size=(1, num_features))
        
    def __call__(self, x):
        if self.training:

            dim = None
            if x.dim() == 2:
                dim = 0
            elif x.dim() == 3:
                dim = (0,1)
            
            mean = x.mean(dim=dim, keepdim=True)
            var = x.var(dim=dim, keepdim=True)
            
            with torch.no_grad():            
                self.running_mean = (1-self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1-self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
        

        x_norm = (x-mean)/torch.sqrt(var + self.epsilon)
        self.out = self.bngain * x_norm + self.bnbias
        
        return self.out
    
    def parameters(self):
        return [self.bnbias, self.bngain]
    
class Tanh:
    def __init__(self):
        pass
    
    def __call__(self, x):
        self.out = torch.tanh(x)
        return self.out
    
    def parameters(self):
        return []
    
    
class Embedding:
    def __init__(self, num_embeddings, embedding_dim):
        self.C = torch.randn((num_embeddings, embedding_dim))
        
    def __call__(self, x):
        self.out = self.C[x]
        return self.out
    
    def parameters(self):
        return [self.C]
    
class FlattenConsecutive:
    def __init__(self, n=None):
        self.n = n
    
    def __call__(self, x):
        
        if self.n != None:
            B, T, C = x.shape
            self.out = x.view(B, T//self.n, C * self.n)
            if self.out.shape[1] == 1:
                self.out = torch.squeeze(self.out, dim=1)
        else:
            self.out = x.view(x.shape[0], -1)
        return self.out
    
    def parameters(self):
        return []
            
class Sequential:
    def __init__(self, layers):
        self.layers = layers
        
    def __call__(self, x):
        for layer in self.layers:
            #in_shape = x.shape
            x = layer(x)
            #out_shape = x.shape
            #print(f"layer: {layer.__class__.__name__} {in_shape}-->{out_shape}")
        self.out = x
        return self.out
    
    def parameters(self):
        return [params for layer in self.layers for params in layer.parameters()]
    
    def train(self):
        for layer in self.layers:
            layer.training = True
    
    def eval(self):
        for layer in self.layers:
            layer.training = False
        
model = Sequential([
    Embedding(nchars, 10),
    FlattenConsecutive(),
    Linear(nchars*10, 100),
    BatchNorm1D(100),
    Tanh()
])
nparams = sum([p.numel() for p in model.parameters()])

# %%
############# Wavenet model

torch.manual_seed(MANUAL_SEED)
emb_size = 10
n_hidden = 100
bn_momentum = 0.1

""" Plain model
model = Sequential([
    Embedding(nchars, emb_size),
    FlattenConsecutive(),
    Linear(block_size*emb_size, n_hidden, bias=False),
    BatchNorm1D(n_hidden),
    Tanh(),
    Linear(n_hidden, nchars)
])
# for block_size=8, emb_size = 10 and n_hidden = 100
# Total params: 11197 (11k)
# train_loss=tensor(1.9706), dev_loss=tensor(2.2834) and test_loss=tensor(2.3076)
# Predicted names: anueda, viva, marian, danis, naysia, alesa, joshlen, lenna, calone, jamil
"""
#"""
model = Sequential([
    Embedding(nchars, emb_size),
    FlattenConsecutive(2), Linear(2*emb_size, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    FlattenConsecutive(2), Linear(2*n_hidden, n_hidden, bias=False), BatchNorm1D(n_hidden), Tanh(),
    Linear(n_hidden, nchars)
])
# for block_size=8, emb_size = 10 and n_hidden = 100
# Total params: 45597 (45k)
# train_loss=tensor(1.8152), dev_loss=tensor(2.2424) and test_loss=tensor(2.2839)
# Predicted names: anugel, vive, marian, danissa, mika, alessa, mahleigh, sepiah, aden, dazi


# for block_size=8, emb_size = 27 and n_hidden = 200
# Total params: 178156 (178k)
# train_loss=tensor(1.6860), dev_loss=tensor(2.2603) and test_loss=tensor(2.3077)
# Predicted names: anuger, viva, marian, daniella, silayah, aremah, lanae, epica, nasiyah, makynzi
#"""

nparams = sum([p.numel() for p in model.parameters()])
print(f"Total parameters: {nparams}")

with torch.no_grad():
    model.layers[-1].w *= 0.1
    for layer in model.layers:
        if isinstance(layer, Linear):
            gain = 5/3
            layer.w *= (gain)

params = model.parameters()
for param in params:
    param.requires_grad = True
    
epochs = 200000
batch_size = 32
loss_fn = CrossEntropyLoss()
losses = []
initial_lr = 0.1
loss_step_count = 10000

model.train()

for epoch in range(epochs):
    
    for param in params:
        param.grad = None
    
    idx = torch.randint(low=0, high=xtrain.shape[0], size=(batch_size,))
    xdata = xtrain[idx]
    ydata = ytrain[idx]

    logits = model(xdata)
    
    loss = loss_fn(logits, ydata)
    loss.backward()

    if epoch%loss_step_count == 0 or epoch == epochs-1:
        loss_p = sum(losses[-loss_step_count:])/loss_step_count if epoch > 0 else loss.item()
        if epoch != 0 and epoch != epochs-1:
            print(f"Epoch: {epoch:6d}/{epochs:5d} - Loss: {loss_p:.4f}")
        elif epoch == epochs-1:
            print(f"Epoch: {epochs:6d}/{epochs:5d} - Loss: {loss_p:.4f}")
        else:
            print(f"Initial Loss: {loss_p:.4f}")
    
    losses.append(loss.item())
    lr = initial_lr if epoch < 100000 else 0.01
    
    ud_temp = []
    with torch.no_grad():
        for param in params:
            #print(param.shape, param.grad.shape)
            param.data += -lr * param.grad

    #if epoch == 1000:
    #break
# %%
plt.plot(torch.tensor(losses).view(-1, 1000).mean(dim=1))

# %%
@torch.no_grad()
def get_loss(data, target):
    logits = model(data)
    loss = loss_fn(logits, target)
    return loss

model.eval()
train_loss = get_loss(xtrain, ytrain)
dev_loss = get_loss(xdev, ydev)
test_loss = get_loss(xtest, ytest)
print(f"{train_loss=}, {dev_loss=} and {test_loss=}") 
# %%

# Prediction
@torch.no_grad()
def predict(total):
    torch.manual_seed(MANUAL_SEED)
    model.eval()
    
    for i in range(total):
        out = []
        context = [eow] * (block_size)
        while True:
            idx = [stoi[c] for c in context[-block_size:]]
            #print(torch.tensor(idx))
            #print(torch.tensor(idx).unsqueeze(dim=0).shape)
            pred = model(torch.tensor(idx).unsqueeze(dim=0))
            
            probs = torch.softmax(pred, dim=1)
            idx = torch.multinomial(probs, 1, True).item()
            if idx == 0:
                break
            
            out.append(itos[idx])
            context.append(itos[idx])
        
        print("".join(out))

predict(10)
# %%

# %%
import torch
from torch.nn import CrossEntropyLoss
import matplotlib.pyplot as plt
from pathlib import Path

# %%
############ DATALOAD AN PREREQUISITE
SAVE_IMAGE = True
MANUAL_SEED = 42
IMAGE_DIR = Path(__file__).resolve().parent.parent / "Notes" / "Month 4"

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

class Linear:
    def __init__(self, in_features, out_features, bias=True, generator=None):
        self.fan_in = in_features
        self.w = torch.randn(size=(in_features, out_features), generator=generator) #/ in_features ** 0.5
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
            mean = x.mean(dim=0, keepdim=True)
            var = x.var(dim=0, keepdim=True)            
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
    
    
# %%

# Baseline model check with classes

g = torch.Generator().manual_seed(MANUAL_SEED)
emb_size = 10
n_hidden = 100
bn_momentum = 0.1

C = torch.randn(size=(nchars, emb_size), requires_grad=True, generator=g)
layers = [
    Linear(emb_size*block_size, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, n_hidden, bias=False, generator=g), 
    BatchNorm1D(n_hidden, momentum=bn_momentum), 
    Tanh(),
    
    Linear(n_hidden, nchars, generator=g), 
    BatchNorm1D(nchars, momentum=bn_momentum)
]

params = [C] + [param for layer in layers for param in layer.parameters()]
n_params = sum([p.numel() for p in params])
print(f"Total Params: {n_params}")

PART = "FinalRun"
initial_lr = 0.1

def save_plot(tool_number):
    if SAVE_IMAGE:
        IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        plt.savefig(IMAGE_DIR / f"Part {PART} Tool {tool_number}.png")

with torch.no_grad():
    layers[-1].bngain *= 0.1
    for layer in layers:
        if isinstance(layer, Linear):
            gain = 5/3
            layer.w *= (gain)/layer.fan_in**0.5 #0.1 #5/3 #setting gain for init

for param in params:
    param.requires_grad = True

epochs = 200000
batch_size = 32
loss_fn = CrossEntropyLoss()
losses = []

for layer in layers:
    layer.training = True

ud_tracking = []

for epoch in range(epochs):
    
    for param in params:
        param.grad = None
    
    idx = torch.randint(low=0, high=xtrain.shape[0], size=(batch_size,), generator=g)
    xdata = xtrain[idx]
    ydata = ytrain[idx]

    embed = C[xdata].view(-1, block_size*emb_size)
    
    logits = embed
    for layer in layers:
        logits = layer(logits)
    
    loss = loss_fn(logits, ydata)
    
    for layer in layers:
        layer.out.retain_grad()
    
    loss.backward()

    if epoch%10000 == 0 or epoch == epochs-1:
        loss_p = sum(losses[-10000:])/10000 if epoch > 0 else loss.item()
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
            
            # udtracking
            ratio = ((lr * param.grad).std() / param.data.std()).log10().item()
            ud_temp.append(ratio)
        ud_tracking.append(ud_temp)

    #if epoch == 1000:
    #    break
    
# %%

# TOOL 1: FORWARD-PASS DIAGNOSTIC — ACTIVATION DISTRIBUTION PER Tanh LAYER
"""
What it shows:
    Histogram of each Tanh layer's output (layer.out) from the last forward
    pass, plus mean, std and % saturation (|out| > 0.97).

How to read it:
    - Healthy: std roughly constant across layers, histograms similar in shape,
      low saturation (a few %).
    - Shrinking std / mass piling at 0  -> activations are vanishing (init gain too small).
    - Mass piled at -1 and +1 / high saturation -> tanh is saturated (init gain too large);
      gradients through saturated units are ~0, so those neurons stop learning.

What changes it:
    Weight init scale (gain), and the presence of BatchNorm. Compare runs
    with different settings and watch how these statistics shift.

Requires:
    Each Tanh layer must store its output (self.out = ...) during forward.
"""

plt.figure(figsize=(10,8))
for idx, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        data = layer.out
        hist, edges = torch.histogram(data, density=True, bins=50)
        
        layername = layer.__class__.__name__        
        mean, std = data.mean(), data.std()
        saturated = (data.abs() > 0.97).float().mean()*100
        
        plt.plot(edges[:-1].detach(), hist.detach(), label=f"Layer {idx} {layername}")
        print(f"Layer {idx:2d}/{layername} >> mean: {mean:.5f}, std: {std:.5f} with {saturated:.2f}% saturation")
        
plt.title("activation distribution")
plt.legend()
save_plot(1)
plt.show()
plt.close()

"""
Layer  2/Tanh >> mean: 0.01100, std: 0.72478 with 14.19% saturation
Layer  5/Tanh >> mean: 0.00313, std: 0.74693 with 17.72% saturation
Layer  8/Tanh >> mean: 0.00981, std: 0.75059 with 16.78% saturation
Layer 11/Tanh >> mean: 0.03041, std: 0.75040 with 16.09% saturation
Layer 14/Tanh >> mean: -0.00930, std: 0.75287 with 15.53% saturation
Layer 17/Tanh >> mean: -0.01438, std: 0.75377 with 15.00% saturation
"""


# TOOL 2: BACKWARD-PASS DIAGNOSTIC — GRADIENT DISTRIBUTION PER Tanh LAYER
"""
What it shows:
    Histogram of the gradient flowing into each Tanh layer's output
    (layer.out.grad) from the last backward pass, plus mean and std.

How to read it:
    - Healthy: gradient std is similar across layers, so every layer
      learns at a comparable rate.
    - Std shrinking toward early layers -> vanishing gradients.
    - Std growing toward early layers   -> exploding gradients.
    Compare with Tool 1: saturated activations are the usual cause of
    vanishing gradients.

Requires:
    layer.out.retain_grad() must be called in forward, and this must be
    run after loss.backward() (otherwise .grad is None).
"""

plt.figure(figsize=(10,8))
for idx, layer in enumerate(layers):
    if isinstance(layer, Tanh):
        data = layer.out.grad
        
        hist, edges = torch.histogram(data, density=True, bins=50)
        layername = layer.__class__.__name__
        mean, std = data.mean(), data.std()
        
        plt.plot(edges[:-1].detach(), hist.detach(), label=f"Layer {idx} {layername}")
        print(f"Layer {idx:2d}/{layername} >> mean: {mean:.5f}, std: {std:.5f}")
        
plt.title("out.grad distribution")
plt.legend()
save_plot(2)
plt.show()
plt.close()

"""
Layer  2/Tanh >> mean: -0.00000, std: 0.00375
Layer  5/Tanh >> mean: 0.00000, std: 0.00350
Layer  8/Tanh >> mean: 0.00000, std: 0.00349
Layer 11/Tanh >> mean: 0.00000, std: 0.00351
Layer 14/Tanh >> mean: -0.00000, std: 0.00342
Layer 17/Tanh >> mean: 0.00000, std: 0.00333
"""


# TOOL 3: BACKWARD-PASS DIAGNOSTIC — WEIGHT GRADIENT DISTRIBUTION PER Linear LAYER
"""
What it shows:
    Histogram of W.grad for each Linear layer (biases skipped), plus a table of
    grad mean, parameter std, and the grad/param ratio (std of W.grad / std of W).

How to read it:
    - The ratio is the key number: it approximates the relative size of one
      update step, lr * ratio. A healthy value is roughly 1e-3 (the exact scale
      depends on lr).
    - Ratio far below that for a layer -> that layer barely learns.
    - Ratio far above that -> its updates are large and may destabilise training.
    - Layers should have similar ratios and similar histogram widths; a
      layer that is an outlier (e.g. the last layer, when its output is scaled
      down at init) is where the imbalance is.
    - Mean should sit near 0 for every layer.

Difference from Tool 2:
    Tool 2 looks at gradients w.r.t. activations (signal flowing through the
    network); this looks at gradients w.r.t. the weights, which is what the
    optimizer actually uses to update.

Requires:
    Run after loss.backward() so W.grad is populated.
    layer.parameters()[0] must be the weight matrix.
"""

print("Layer\tW.grad.mean\tW.grad.std\tgrad/param ratio")
plt.figure(figsize=(10,8))
for idx, layer in enumerate(layers):
    if isinstance(layer, Linear):
        wparam = layer.parameters()[0] #taking only the "w" parameter
        
        grad_mean = wparam.grad.mean()
        grad_std = wparam.grad.std()
        param_std = wparam.std()
        grad_to_param_ratio = grad_std / param_std
        
        class_name = layer.__class__.__name__
        
        hist, edges = torch.histogram(wparam.grad, density=True)
        plt.plot(edges[:-1].detach(), hist.detach(), label=f"Grad of Layer {idx}-{class_name} W")
        
        print(f"{idx:2d}\t{grad_mean:+.5f}\t{grad_std:.5f}\t\t{grad_to_param_ratio:.5f}")
        
plt.legend()
plt.title("param.grad distribution of Ws in linear layers")
save_plot(3)
plt.show()
plt.close()


"""
Layer	W.grad.mean	W.grad.std	grad/param ratio
 0	-0.00013	0.00767		0.01821
 3	-0.00004	0.00520		0.02037
 6	-0.00008	0.00514		0.02045
 9	+0.00003	0.00521		0.02119
12	-0.00010	0.00529		0.02191
15	-0.00000	0.00497		0.02127
18	-0.00010	0.00769		0.02318
"""


# TOOL 4: TRAINING-TIME DIAGNOSTIC — UPDATE-TO-PARAMETER RATIO OVER TIME
"""
What it shows:
    For every weight matrix, the log10 of (std of the update lr * grad) / (std of
    the parameter), recorded at every training step in ud_tracking. One line per
    parameter, x-axis = step. The horizontal line marks the -3 reference (1e-3).

How to read it:
    - Rule of thumb: each update should change a weight by about 1/1000 of its
      scale, so lines should hover around -3.
    - Lines well below -3 -> learning too slowly (lr too low, or that layer's
      gradients are tiny).
    - Lines well above -3 -> updates too large, risk of instability.
    - Lines should be roughly parallel and stable; a layer drifting away
      from the rest is under- or over-trained relative to the others.
    - Early steps are noisy while the network settles; judge the trend.
    - An lr drop (e.g. 0.1 -> 0.01) shows as a downward step of exactly 1.

Difference from Tool 3:
    Tool 3 is a single snapshot of grad/param at the last step. This tracks
    the actual update (including lr) across the whole run.

Requires:
    ud_tracking filled inside the training loop, one list of log10 ratios per
    step, in the same order as params. Biases/gains (any param with a dimension of
    size 1) are skipped by the plotting condition.
"""

x_axis = torch.arange(0, len(ud_tracking))
plt.figure(figsize=(10,8))
for idx, param in enumerate(params):
    if param.shape[0] != 1 and param.shape[1] != 1:
        pshape = param.shape
        ud_data = [ud_tracking[i][idx] for i in range(len(ud_tracking))]
        label = f"param {idx}, shape {pshape}"
        
        plt.plot(x_axis, ud_data, label=label)
plt.plot([0, len(ud_tracking)], [-3, -3])
plt.legend()
plt.title("grad/param update ratio for W params")
save_plot(4)
plt.show()
plt.close()
# %%
@torch.no_grad()
def get_loss(data, target):
    embed = C[data].view(-1, block_size*emb_size)
    logits = embed
    for layer in layers:
        logits = layer(logits)
    loss = loss_fn(logits, target)
    return loss

for layer in layers:
    layer.training = False
    
train_loss = get_loss(xtrain, ytrain)
dev_loss = get_loss(xdev, ydev)
test_loss = get_loss(xtest, ytest)
print(f"{train_loss=}, {dev_loss=} and {test_loss=}") 

# train_loss=tensor(1.7748), dev_loss=tensor(2.2350) and test_loss=tensor(2.2708)

# %%
g = torch.Generator().manual_seed(MANUAL_SEED)

for layer in layers:
    layer.training = False

# Prediction
@torch.no_grad()
def predict(total):
    for i in range(total):
        out = []
        context = [eow] * (block_size)
        while True:
            idx = [stoi[c] for c in context[-3:]]
            
            embed = C[idx].view(-1, emb_size * block_size)
            pred = embed
            for layer in layers:
                pred = layer(pred)
            
            probs = torch.softmax(pred, dim=1)
            idx = torch.multinomial(probs, 1, True, generator=g).item()
            if idx == 0:
                break
            
            out.append(itos[idx])
            context.append(itos[idx])
        
        print("".join(out))

predict(10)

"""
ayusufa
tia
aarian
hanvitha
illa
yna
otman
amonn
easachana
immer
"""

# %%

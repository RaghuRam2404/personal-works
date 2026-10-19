import numpy as np
import math

##########

a = 3
b = 10
c = 12
d = 4
expr = ((a+b)**2)+(a*c)/d+d+b/2
print(expr)

def valuate_expr(a,b,c,d, ah, bh, ch, dh):
    return (((a+ah)+(b+bh))**2)+((a+ah)*(c+ch))/(d+dh)+(d+dh)+(b+bh)/2

def valuate_derivative(d1, d2, h):
    return (d1-d2)/(2*h)

ah = .00000001
bh = .00000001
ch = .00000001
dh = .00000001

expr1 = valuate_expr(a, b, c, d, ah, 0, 0, 0)
expr2 = valuate_expr(a, b, c, d, -ah, 0, 0, 0)
da = valuate_derivative(expr1, expr2, ah)
print(f"estimated da: {da}")
da = (2*(a+b))+c/d
print(f"real da: {da}")

expr1 = valuate_expr(a, b, c, d, 0, +bh, 0, 0)
expr2 = valuate_expr(a, b, c, d, 0, -bh, 0, 0)
db = valuate_derivative(expr1, expr2, bh)
print(f"estimated db: {db}")
db = (2*(a+b))+(1/2)
print(f"real db: {db}")

expr1 = valuate_expr(a, b, c, d, 0, 0, +ch, 0)
expr2 = valuate_expr(a, b, c, d, 0, 0, -ch, 0)
dc = valuate_derivative(expr1, expr2, ch)
print(f"estimated dc: {dc}")
dc = (a/d)
print(f"real dc: {dc}")

expr1 = valuate_expr(a, b, c, d, 0, 0, 0, dh)
expr2 = valuate_expr(a, b, c, d, 0, 0, 0, -dh)
dd = valuate_derivative(expr1, expr2, dh)
print(f"estimated dd: {dd}")
dd = (-1)*(a*c)*(d**-2)+1
print(f"real dd: {dd}")

##########

class Value:
    
    def __init__(self, value, name="", children=()):
        self.value = value
        self.name = name
        self.grad = 0.0
        self.children = children
        self._backward = lambda: None
        
    def _clean(self, value):
        if isinstance(value, (int, float)):
            value = Value(value)
        return value
            
    
    def __add__(self, other):
        
        other = self._clean(other)
        out = Value(self.value + other.value, children=(self, other))
        
        def backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
            #print("__add__", type(self.grad), type(other.grad))
        
        out._backward = backward
        return out
            
    def __mul__(self, other):
        
        other = self._clean(other)
        out = Value(self.value * other.value, children=(self, other))
        
        def backward():
            self.grad += other.value * out.grad
            other.grad += self.value * out.grad
            #print("__mul__", type(self.grad), type(other.grad))
        
        out._backward = backward
        return out
            
    def __sub__(self, other):
        
        other = self._clean(other)
        #a-b = a + (b*-1)
        return self.__add__(other * -1)
    
    def __radd__(self, other):
        return other.__add__(self)
    
    def __rmul__(self, other):
        return other.__mul__(self)
    
    def __rsub__(self, other):
        return other.__sub__(self)
    
    def __truediv__(self, other):
        other = self._clean(other)
        #a/b = a*b**-1
        return self.__mul__(other**-1)

    def tanh(self):
        x = self.value
        tanh = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(tanh,children=(self,))
        
        def backward():
            self.grad += (1 - tanh**2) * out.grad
            #print(type(self.grad))
        
        out._backward = backward
        return out
    
    def __pow__(self, other):
        
        other = self._clean(other)
        out = Value(self.value ** other.value, children=(self, other))
        
        def backward():
            self.grad += other.value * (self.value ** (other.value-1)) * out.grad
            #print("__pow__", type(self.grad))
        out._backward = backward
        
        return out
    
    def backward(self):
        self.grad = 1.0
        visited = set()
        nodes = []
        def topo(node):
            if node not in visited:
                visited.add(node)
                for child in node.children:
                    topo(child)
                nodes.append(node)
        
        topo(self)
        
        nodes = reversed(nodes)
        for node in nodes:
            #print(node)
            node._backward()
        
    def __format__(self):
        return self.value
    
    def __repr__(self):
        string_rep = ""
        if self.name is not None and self.name != "":
            string_rep += f"{self.name}: "
        string_rep += str(self.value)
        return string_rep
 
a = Value(3)
b = Value(10)
c = Value(12)
d = Value(4)
expr = ((a+b)**2)+(a*c)/d+d+b/2
expr.backward()
print(f"{a.grad=}, {b.grad=}, {c.grad=}, {d.grad=}")

a = Value(0.3,"a")
b = Value(0.10,"b")
c = Value(0.12,"c")
d = Value(0.4,"d")
expr = ((a+b)**2)+(a*c)/d+d+b/2
e = expr.tanh(); e.name = 'e'
e.backward()
print(f"a.grad={a.grad:.4f}, a.grad={b.grad:.4f}, c.grad={c.grad:.4f}, d.grad={d.grad:.4f}, e.grad={e.grad:.4f}")

##########

class Neuron:
    
    def __init__(self, nin):
        #nin -> No of inputs neurons
        self.nin = nin
        self.w = [Value(val) for val in np.random.uniform(-1, 1, size=nin)]
        self.b = Value(np.random.uniform(-1,1))
    
    def __call__(self, x):
        # tanh(wx+b)
        assert len(self.w) == len(x), f"len(self.w):{len(self.w)} mismatches len(x):{len(x)}"
        wxes = [wi*xi for wi, xi in zip(self.w, x)]
        wx = sum(wxes)
        wxb = wx + self.b
        a = wxb.tanh()
        return a
    
    def parameters(self):
        return self.w + [self.b]
    
neuron = Neuron(4)
neuron.w, neuron.b
x = [val for val in np.random.uniform(-1, 1, size=4)]
neuron(x), neuron.parameters()
neuron(x).backward()
neuron.w[0].grad, neuron.b.grad

###########

class Layer:
    
    def __init__(self, fin, fout):
        self.fin = fin
        self.fout = fout
        # I need fout neurons with fin as input to each neuron
        self.neurons = [Neuron(fin) for _ in range(fout)]
    
    def __call__(self, x):
        return [neuron(x)  for neuron in self.neurons]
    
    def parameters(self):
        return [ p for neuron in self.neurons for p in neuron.parameters() ]

layer = Layer(4,4)
layer.parameters()
out = layer(x)
for o in out:
    print(o)
    
###########

class MLP:
    
    def __init__(self, layers):
        self.layers = []
        for lcount in range(len(layers)-1):
            #print(layers[lcount], layers[lcount+1])
            self.layers.append(Layer(layers[lcount], layers[lcount+1]))
        
    def __call__(self, x):
        out = x
        for layer in self.layers:
            lin, lout = len(out), 0
            out = layer(out)
            lout = len(out)
            #print(f"{lin=}, {lout=}, {layer.fin=}, {layer.fout=}")
        return out[0] if len(out) == 1 else out
    
    def zero_grad(self):
        params = self.parameters()
        for param in params:
            param.grad = 0.0
    
    def parameters(self):
        params = [params for layer in self.layers for params in layer.parameters()]
        return params
    

x = [val for val in np.random.uniform(-1, 1, size=3)]
print(f"Input Size: {len(x)}")
network = MLP([len(x),4,4,1])
print(network(x))
print(network.parameters())
print(len(network.parameters()))

###########

xn = 5
features = 4
xs = np.random.randn(xn, features)
ys = np.random.randn(xn)
print(xs, ys)

model = MLP([features, 4, 4, 1])
epochs = 20

for i in range(epochs):
    model.zero_grad()
    ypred = [model(x) for x in xs]
    loss = sum([(yp-yt)**2 for yp, yt in zip(ypred, ys) ])
    print(loss)
    
    loss.backward()
    
    for p in model.parameters():
        p.value += -0.01*p.grad
        

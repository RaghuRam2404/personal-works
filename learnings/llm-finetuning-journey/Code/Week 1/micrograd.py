import numpy as np
import math
import matplotlib.pyplot as plt
%matplotlib inline


# basic derivative estimation check
a = 2
b = 4
c = a*b
print(c)

h = 0.0001
a = a + h
b = 4
c1 = a*b
print(c1)

derivate = (c1-c)/h
print(derivate) #print 4



# Value holder

class Value:
    
    def __init__(self, data, label="", op="", children=()):
        self.data = data
        self.label = label
        self._op = op
        self._prev = set(children)

        self.grad = 0.0
        self._backward = lambda: None
        
    def __add__(self, data, label=""):
        out = Value(self.data + data.data, label, op="+", children=(self, data))
        def backward():
            self.grad += 1.0 * out.grad
            data.grad += 1.0 * out.grad        
        self._backward = backward
        
        return out
    
    def __mul__(self, data, label=""):
        out = Value(self.data * data.data, label, op="*", children=(self, data))
        def backward():
            self.grad += out.grad * data.data
            data.grad += out.grad * self.data
        self._backward = backward
        
        return out
    
    def tanh(self, label=""):
        x = self.data
        tanh = (math.exp(2*x)-1)/(math.exp(2*x)+1)
        out = Value(tanh, op='tanh', children=(self,))
        
        def backward():
            self.grad += (1 - tanh**2) * out.grad
        
        self._backward = backward
        
        return out
    
    
    def backward(self):
        self.grad = 1.0

        def topo(node):
            nodes = []
            visited = set()
            nodes.append(node)
            visited.add(node)
            i = 0
            while True:
                children = node._prev
                for n in children:
                    if n not in visited:
                        nodes.append(n)
                if (i+1) >= len(nodes):
                    break
                i+=1
                node = nodes[i]
            return nodes
        
        nodes = topo(self) #sorted ones in the nodes variable
        
        for n in nodes:
            n._backward()
            #print(n, n.grad)
        
    def __repr__(self):
        return f"Value of {self.label}: {self.data}"



a = Value(2, label='a')
b = Value(4, label='b')
print(a, b)
mul = a * b
mul.label = "a*b"
add = a + b
add.label = "a+b"
print(mul, add)

# Forward pass:
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

w1x1 = x1 * w1; w1x1.label='w1x1'
w2x2 = w2 * x2; w2x2.label='w2x2'
x1w1w2x2 = w1x1+w2x2; x1w1w2x2.label='x1w1w2x2'
n = x1w1w2x2 + b; n.label = 'n'

o = n.tanh(); o.label='o'

print(n, o)

# Backward pass
o.grad = 1.0
o._backward()

n._backward()

x1w1w2x2._backward()
b._backward()

w2x2._backward()
w1x1._backward()

w1._backward()
x1._backward()
w2._backward()
x2._backward()

print(f"w1.grad: {w1.grad}, x1.grad: {x1.grad}")
print(f"w2.grad: {w2.grad}, x2.grad: {x2.grad}")

# Backward pass in a single function
o.backward()
print(f"w1.grad: {w1.grad}, x1.grad: {x1.grad}")
print(f"w2.grad: {w2.grad}, x2.grad: {x2.grad}")

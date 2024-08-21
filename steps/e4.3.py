import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output
    
    def forward(self, x):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

def number_diff(f, x, esp=1e-4):
    x0 = Variable(x.data - esp)
    x1 = Variable(x.data + esp)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * esp)


def derivation_of_f(x):
    return 4 * x.data * np.exp(2 * (x.data ** 2))

def f(x):
    A  = Square()    
    B = Exp()    
    C = Square()    
    return C(B(A(x)))

x = Variable(np.array(0.5))
y0 = number_diff(f, x)
print("y0", y0)

y1 = derivation_of_f(x)
print("y1:", y1)


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

f = Square()
x = Variable(np.array(2.0))
y = number_diff(f, x)
print(y)



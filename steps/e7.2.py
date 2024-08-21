import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None
        self.creator = None


    def set_creator(self, c):
        self.creator = c

class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        output.set_creator(self)
        self.output = output
        return output
    
    def forward(self, x):
        raise NotImplementedError()
    
    def backward(self, gy):
        raise NotImplementedError()
    

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


def derivation_of_f(x):
    return 4 * x.data * np.exp(2 * (x.data ** 2))

x = Variable(np.array(0.5))
A  = Square()    
B = Exp()    
C = Square()   

a = A(x)
b = B(a)
y = C(b)

def f(x):
    return C(B(A(x)))


y0 = derivation_of_f(x)
print("y0\t", y0)


y.grad = np.array(1.0)
fc = y.creator
b.grad = fc.backward(y.grad)
fb = b.creator
a.grad = fb.backward(b.grad)

fa = a.creator
x.grad = fa.backward(a.grad)

print("x.grad\t", x.grad)


import numpy as np


class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None

class Function:
    def __call__(self, input):
        self.input = input
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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

def number_diff(f, x, esp=1e-4):
    x0 = Variable(x.data - esp)
    x1 = Variable(x.data + esp)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * esp)


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

# print("y@nd\t", number_diff(f, x)) #!!! 3.298761782256191 != 3.297442541400256

y.grad = np.array(1.0)
b.grad = C.backward(y.grad)
a.grad = B.backward(b.grad)
x.grad = A.backward(a.grad)

print("x.grad\t", x.grad)


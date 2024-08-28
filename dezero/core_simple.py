import numpy as np
import heapq
import weakref

class Variable:
    __array_priority = 200 # useless for Python3.9

    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    @property
    def shape(self):
        return self.data.shape
    
    @property
    def ndim(self):
        return self.data.ndim
    
    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype
    
    def __neg__(self):
        return neg(self)
    
    def __sub__(self, other):
        return sub(self, other)
    
    # non-var - var(self)
    def __rsub__(self, other):
        return rsub(self, other)

    def __mul__(self, other):
        return mul(self, other)
    
    def __truediv__(self, other):
        return div(self, other)
    
    def __rtruediv__(self, other):
        return rdiv(self, other)
    
    def __pow__(self, other):
        return pow(self, other)

    def __add__(self, other):
        return add(self, other)
    
    def __rmul__(self, other):
        return mul(self, other)
    
    def __radd__(self, other):
        return add(self, other)    
    
    def __repr__(self):
        if self.data is None:
            return 'varible(None)'
        else:
            c = str(self.data).replace('\n', '\n'+ ' ' * 9)
            return f'variable({c})'
    
    def __len__(self):
        return len(self.data)
    
    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def reset_grad(self):
        self.grad = None

    def backward(self, retain_grad = False):
        def pop_func(alist):
            return heapq.heappop(alist)
        
        def append_func(alist, f):
            heapq.heappush(alist, f)

        if self.grad is None:
            self.grad = np.ones_like(self.data)
        
        fs = [self.creator]            
        while fs:
            f = pop_func(fs)
            ygs = [o().grad for o in f.outputs]
            xgs = f.backward(*ygs)
            if not isinstance(xgs, tuple):
                xgs = (xgs,)
    
            for x, g in zip(f.inputs, xgs):
                if x.grad is None:
                    x.grad = g
                    if x.creator is not None: # append the function once when the var: x first appeared
                        append_func(fs, x.creator)
                else: # Var:x, is repeated
                    x.grad = x.grad + g
            
            # for the very begin of variables, their backward codes Do NOT reach here, because they don't have any creator
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None # y is a weakref of the output

def as_variable(x):
    if isinstance(x, Variable):
        return x
    else:
        return Variable(x)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Config:
    enable_backprop = True # backward propagate

class Function:
    def __init__(self) -> None:
        self.generation = 0
    
    def __lt__(self, other):
            return -self.generation < -other.generation
    
    def __call__(self, *inputs): # input:[x0, x1, ...]@Variable -> [y0, y1, ...]@Variable
        self.inputs = [as_variable(x) for x in inputs]
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) #unwrap
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in self.inputs])
            for output in outputs:
                output.set_creator(self)
            self.outputs = [weakref.ref(o) for o in outputs]

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx

class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.inputs[0].data
        gx = np.exp(x) * gy
        return gx
    
class Pow(Function):
    def __init__(self, exp):
        self.exp = exp

    def forward(self, x):
        return x ** self.exp
        
    # x^e
    def backward(self, gy):
        x, exp = self.inputs[0].data, self.exp
        return gy * (exp * (x ** (exp-1)))

class Neg(Function):
    def forward(self, x):
        return -x
    
    def backward(self, gy):
        return -gy

class Mul(Function):
    def forward(self, a, b):
        return a * b
    
    def backward(self, gy):
        a, b = self.inputs[0].data, self.inputs[-1].data
        return gy * b, gy * a
    
class Div(Function):
    def forward(self, a, b):
        return a / b
    
    # a / b <==> a * (1/b) <==> a * (b^-1)
    # y = a * u
    # u = 1 / b = b^ -1
    # 'y/'b = 'y/'u * ('u/'b) = a * (-1 * b ** -2)
    def backward(self, gy):
        a, b = self.inputs[0].data, self.inputs[-1].data
        return gy * 1 / b, gy * a * (-1) * (b ** -2)

class Add(Function):
    def forward(self, a, b):
        y = a + b
        return (y,)
    
    def backward(self, gy):
        return gy, gy
    
class Sub(Function):
    def forward(self, a, b):
        return a - b
    
    # a - b <==> a + (-b)
    def backward(self, gy):
        return gy, -gy

class Identical(Function):
    def forward(self, x):
        return x
    
    def backward(self, gy):
        return gy # ??? 1 or gy

def square(x):
  return Square()(x)

def exp(x):
  return Exp()(x)

def pow(x, exp):
  return Pow(exp)(x)

def add(x0, x1):
  return Add()(x0, as_array(x1))

def identical(x):
  return Identical()(x)

def mul(x0, x1):
  return Mul()(x0, as_array(x1))

def neg(x):
  return Neg()(x)

def sub(x0, x1):
  return Sub()(x0, as_array(x1))

# x1 - x0
def rsub(x0, x1):
  x = as_array(x1)
  return Sub()(x, x0)

def div(x0, x1):
  return Div()(x0, as_array(x1))

# x1 / x0
def rdiv(x0, x1):
  x = as_array(x1)
  return Div()(x, x0)

import contextlib

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config( 'enable_backprop' , False)

with no_grad():
    x =Variable(np.array(2.0))
    y = square(x)

def setup_variable():
    pass
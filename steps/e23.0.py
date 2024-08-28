import sys
sys.path.append('../')

from dezero import Variable
import numpy as np

x = Variable(np.array([3]))
print(x)

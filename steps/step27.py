import math

import numpy as np

from dezero import Function
from dezero import Variable


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0]
        gx = np.cos(x) * gy
        return gx


def sin(x):
    return Sin()(x)


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t

        # t为第i次加的项，当这个项足够小时，break
        if abs(t.data) < threshold:
            break
    return y


x = Variable(np.array(np.pi / 4))
y = sin(x)
# y = my_sin(x)
y.backward(create_graph=True)

print(y.data)
print(x.grad)

import numpy as np
from dezero import Variable


def f(x):
    y = x ** 4 - 2 * x ** 2
    return y


x = Variable(np.array(2.0))
iters = 10

for i in range(iters):
    print(i, x)

    y = f(x)
    # 求一阶导前要把梯度清0
    x.cleargrad()
    # 求一阶导
    y.backward(create_graph=True)

    gx = x.grad
    # 再清0，求二阶导
    x.cleargrad()
    gx.backward()
    gx2 = x.grad

    x.data -= gx.data / gx2.data

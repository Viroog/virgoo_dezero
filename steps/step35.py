import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


x = Variable(np.array(1.0))
y = F.tanh(x)
x.name = 'x'
y.name = 'y'
y.backward(create_graph=True)

iters = 4
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx = x.grad
# iters=0求一阶导数，类推
gx.name = 'gx' + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file=f'tanh_{iters+1}.png')
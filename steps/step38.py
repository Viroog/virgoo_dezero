import numpy as np

import dezero.functions as F
from dezero import Variable

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y = F.reshape(x, (6,))
y.backward(retain_grad=True)
print(x.grad)

x.cleargrad()
# y = F.transpose(x)
# y = x.transpose()
y = x.T
y.backward()
print(x.grad)
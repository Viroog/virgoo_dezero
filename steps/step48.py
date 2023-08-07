import math

import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
from dezero.models import MLP
import dezero.functions as F

x, t = dezero.datasets.get_spiral(train=True)

# 可视化
# x_class_0 = [x[i] for i in range(len(x)) if t[i] == 0]
# x_class_1 = [x[i] for i in range(len(x)) if t[i] == 1]
# x_class_2 = [x[i] for i in range(len(x)) if t[i] == 2]
#
# print(x_class_0)
#
# plt.scatter(*zip(*x_class_0), marker='o')
# plt.scatter(*zip(*x_class_1), marker='^')
# plt.scatter(*zip(*x_class_2), marker='x')
# plt.show()

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

model = MLP((hidden_size, 3))
optimizers = optimizers.SGD(lr=lr)
optimizers.setup(model)

data_size = len(x)
# 向上取整
max_iter = math.ceil(data_size / batch_size)

total_loss = []
epochs = []

# 两个循环，一个是epoch的，一个是iter的(即一次小批量处理)
for epoch in range(max_epoch):
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size:(i + 1) * batch_size]
        batch_x = x[batch_index]
        batch_t = t[batch_index]

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizers.update()

        sum_loss += float(loss.data) * len(batch_t)

    avg_loss = sum_loss / data_size
    total_loss.append(avg_loss)
    epochs.append(epoch)
    print(f'epoch {epoch+1}, loss {avg_loss}')


# loss曲线
plt.plot(epochs, total_loss)
plt.show()

# 边界图
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]

with dezero.no_grad():
    score = model(X)
predict_cls = np.argmax(score.data, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)

# Plot data points of the dataset
N, CLS_NUM = 100, 3
markers = ['o', 'x', '^']
colors = ['orange', 'blue', 'green']
for i in range(len(x)):
    c = t[i]
    plt.scatter(x[i][0], x[i][1], s=40,  marker=markers[c], c=colors[c])
plt.show()
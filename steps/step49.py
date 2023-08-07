import math

import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
from dezero.models import MLP
import dezero.functions as F

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizers = optimizers.SGD(lr=lr)
optimizers.setup(model)

data_size = len(train_set)
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
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

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


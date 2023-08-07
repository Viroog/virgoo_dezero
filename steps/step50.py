import math

import numpy as np
import matplotlib.pyplot as plt
import dezero
from dezero import optimizers
from dezero.models import MLP
import dezero.functions as F
from dezero import DataLoader

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral(train=True)
test_set = dezero.datasets.Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizers = optimizers.SGD(lr=lr)
optimizers.setup(model)


# 两个循环，一个是epoch的，一个是iter的(即一次小批量处理)
for epoch in range(max_epoch):

    sum_loss, sum_acc = 0, 0

    # 从DataLoader中取出小批数据
    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)

        model.cleargrads()
        loss.backward()
        optimizers.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    print(f'epoch: {epoch+1}')
    print('train loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    # 测试集，不需要进行反向传播，即不建立计算图，减少内存损耗
    with dezero.no_grad():
        for x, t in test_loader:
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)

            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {:.4f}, accuracy: {:.4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))




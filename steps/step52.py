import time

import cupy
import numpy as np

import dezero
import matplotlib.pyplot as plt
from dezero.models import MLP
from dezero.optimizers import SGD
import dezero.functions as F
from dezero.dataloaders import DataLoader

max_epoch = 5
batch_size = 100

train_set = dezero.datasets.MNIST(train=True)
train_loader = DataLoader(train_set, batch_size)

model = MLP((1000, 10))
optimizer = SGD()
optimizer.setup(model)

if dezero.cuda.gpu_enable:
    # 两个足矣，一个是让输入数据加载为cupy，一个是让模型参数加载为cupy
    train_loader.to_gpu()
    model.to_gpu()

for epoch in range(max_epoch):
    start = time.time()
    sum_loss, sum_acc = 0, 0

    for x, t in train_loader:
        y = model(x)
        loss = F.softmax_cross_entropy(y, t)
        acc = F.accuracy(y, t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)

    elapsed_time = time.time() - start
    print(f'epoch {epoch+1}')
    print('train loss: {:.4f}, accuracy: {:.4f}, time: {:.4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set), elapsed_time))
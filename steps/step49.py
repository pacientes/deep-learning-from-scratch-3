if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import math
import dezero
import numpy as np

from dezero import optimizers
import dezero.functions as F
from dezero.models import MLP

# # dataset 테스트
# train_set = dezero.datasets.Spiral(train=True)
# print(train_set[0])
# print(len(train_set))


# # dataset minibatch
# batch_index = [0, 1, 2]  # 0에서 2번째까지의 데이터를 꺼냄
# batch = [train_set[i] for i in batch_index]

# x = np.array([example[0] for example in batch])
# t = np.array([example[1] for example in batch])

# print(x.shape)
# print(t.shape)


# 데이터 학습

# Hyperparameters
max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = dezero.datasets.Spiral()
model = MLP((hidden_size, 3))
optimizer = optimizers.SGD(lr).setup(model)

data_size = len(train_set)
max_iter = math.ceil(data_size / batch_size)

for epoch in range(max_epoch):
    # Shuffle index for data
    index = np.random.permutation(data_size)
    sum_loss = 0

    for i in range(max_iter):
        batch_index = index[i * batch_size : (i + 1) * batch_size]
        batch = [train_set[i] for i in batch_index]
        batch_x = np.array([example[0] for example in batch])
        batch_t = np.array([example[1] for example in batch])

        y = model(batch_x)
        loss = F.softmax_cross_entropy(y, batch_t)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(batch_t)

    # Print loss every epoch
    avg_loss = sum_loss / data_size
    print("epoch %d, loss %.2f" % (epoch + 1, avg_loss))

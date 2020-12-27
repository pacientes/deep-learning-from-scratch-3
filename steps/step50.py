if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero
import dezero.functions as F

from dezero.datasets import Spiral
from dezero import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD

# batch_size = 10
# max_epoch = 1

# train_set = Spiral(train=True)
# test_set = Spiral(train=False)
# train_loader = DataLoader(train_set, batch_size)
# test_loader = DataLoader(test_set, batch_size, shuffle=False)

# for epoch in range(max_epoch):
#     for x, t in train_loader:
#         print(x.shape, t.shape)  # x, t는 훈련 데이터
#         break

#     # 에포크 끝에서 테스트 데이터를 꺼낸다.
#     for x, t in test_loader:
#         print(x.shape, t.shape)  # x, t는 테스트 데이터
#         break


# accuracy 테스트

# y = np.array([[0.2, 0.8, 0], [0.1, 0.9, 0], [0.8, 0.1, 0.1]])
# t = np.array([1, 2, 0])
# acc = F.accuracy(y, t)
# print(acc)

# 스파이럴 학습

max_epoch = 300
batch_size = 30
hidden_size = 10
lr = 1.0

train_set = Spiral(train=True)
test_set = Spiral(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

model = MLP((hidden_size, 3))
optimizer = SGD(lr).setup(model)

for epoch in range(max_epoch):
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

    print("epoch: {}".format(epoch + 1))
    print(
        "train loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(train_set), sum_acc / len(train_set)
        )
    )

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad():  # 기울기 불필요 모드
        for x, t in test_loader:  # 테스트용 미니배치 데이터
            y = model(x)
            loss = F.softmax_cross_entropy(y, t)
            acc = F.accuracy(y, t)  # 테스트 데이터의 인식 정확도
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print(
        "test loss: {:.4f}, accuracy: {:.4f}".format(
            sum_loss / len(test_set), sum_acc / len(test_set)
        )
    )

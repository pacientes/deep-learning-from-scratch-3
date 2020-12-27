if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero
import dezero.functions as F

from dezero import DataLoader
from dezero.models import MLP
from dezero.optimizers import SGD

# MNIST 데이터 학습하기

max_epoch = 5
batch_size = 100
hidden_size = 1000

# /root/.dezero/t10k-images-idx3-ubyte.gz 경로에 있음
train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation=F.relu)
optimizer = SGD().setup(model)

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

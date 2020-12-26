if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F

# 토이 데이터셋
np.random.seed(0)  # 시드 고정
x = np.random.rand(100, 1)
y = 5 + 2 * x + np.random.rand(100, 1)  # y에 무작위 노이즈 추가
x, y = Variable(x), Variable(y)  # 생략 가능

W = Variable(np.zeros((1, 1)))
b = Variable(np.zeros(1))


def predict(x):
    y = F.matmul(x, W) + b
    return y


lr = 0.1
iters = 100


for i in range(iters):
    y_pred = predict(x)
    loss = F.mean_square_error(y, y_pred)

    W.cleargard()
    b.cleargard()
    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    print(W, b, loss)
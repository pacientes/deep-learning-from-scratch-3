# %% numerical differentiation (bad version)
def numerical_diff(f, x):
    h = 10e-50
    return (f(x + h) - f(x)) / h


# %% numerical differentiation
def numerical_diff(f, x):
    h = 1e-4  # 0.0001
    return (f(x + h) - f(x - h)) / (2 * h)


# %% numerical differentiation example
def function_1(x):
    return 0.01 * x ** 2 + 0.1 * x


# %% plot function_1 graph
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel("x")
plt.ylabel("f(x)")
plt.plot(x, y)
plt.show()


#%% result check
print(numerical_diff(function_1, 5))
print(numerical_diff(function_1, 10))


# %% partial derivative
def function_2(x):
    return np.sum(x ** 2)


# %% x0 = 3, x1 = 4 for function_2 example diff x0
def function_tmp(x0):
    return x0 * x0 + 4.0 ** 2.0


print(numerical_diff(function_tmp, 3.0))


# %% x0 = 3, x1 = 4 for function_2 example diff x1
def function_tmp(x1):
    return 3.0 ** 2.0 + x1 * x1


print(numerical_diff(function_tmp, 4.0))


# %% numerical gradient, Chapter 4.4
def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        # calc f(x+h)
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # calc f(x-h)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2 * h)
        x[idx] = tmp_val  # value restore

    return grad


# %% numerical gradient example
print(numerical_gradient(function_2, np.array([3.0, 4.0])))
print(numerical_gradient(function_2, np.array([0.0, 2.0])))
print(numerical_gradient(function_2, np.array([3.0, 0.0])))


# %% gradient_descent
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad

    return x


# %% sample gradient descent
def function_2(x):
    return x[0] ** 2 + x[1] ** 2


init_x = np.array([-3.0, 4.0])
gradient_descent(f=function_2, init_x=init_x, lr=0.1, step_num=100)
# %% sample gradient descent 2

# big learning rate
init_x = np.array([-3.0, 4.0])
ret = gradient_descent(f=function_2, init_x=init_x, lr=10.0, step_num=100)
print(ret)

# small learning rate
init_x = np.array([-3.0, 4.0])
ret = gradient_descent(f=function_2, init_x=init_x, lr=1e-10, step_num=100)
print(ret)


# %% simpleNet
import sys, os

sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradient import numerical_gradient


class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss


# %% sample simpleNet
net = simpleNet()
print(net.W)  # weight

x = np.array([0.6, 0.9])
p = net.predict(x)
print(p)

maxIdx = np.argmax(p)
print(maxIdx)  # maximum index

# ground truth
t = np.array([0, 0, 1])
net.loss(x, t)
# %%
def f(W):
    return net.loss(x, t)


dW = gradient_descent(f, net.W)
print(dW)
# %% function to lambda

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

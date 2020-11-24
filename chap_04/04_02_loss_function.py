# %% sum of squares for error, SSE
import numpy as np


def sum_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# %% SSE sample
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# example 1
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))

# example 2
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(sum_squares_error(np.array(y), np.array(t)))

# %% cross entropy error, CEE
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))


# %% CEE sample
t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]

# example 1
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# example 2
y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

# %% mini-batch
import sys, os

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname("__file__"))))
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)
print(t_train.shape)

# %% get mnist batch

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)

x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]

# %% numpy random test

print(np.random.choice(60000, 10))

# %% cross entropy error for mini batch
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]
    # for one-hot encoding...
    return -np.sum(t * np.log(y + 1e-7)) / batch_size
    # for normal data... ex 2 or 7
    # return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

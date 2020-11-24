# %% step function
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


# %% step function - simple version
import numpy as np


def step_function(x):
    y = x > 0
    return y.astype(np.int)


x = np.array([-1.0, 1.0, 2.0])
print(x)

y = x > 0
print(y)

# %%

y = y.astype(np.int)
print(y)

# %% plot step function
import matplotlib.pyplot as plt


def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# %% sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# %%
x = np.array([-1.0, 1.0, 2.0])
print(sigmoid(x))

# %% remind broadcast
t = np.array([1.0, 2.0, 3.0])
print(1.0 + t)
print(1.0 / t)

# %% plot sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()

# %% plot step-function and sigmoid function
x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1, label="sigmoid")
plt.plot(x, y2, linestyle="--", label="step function")
plt.ylim(-0.1, 1.1)
plt.show()

# %% relu
def relu(x):
    return np.maximum(0, x)

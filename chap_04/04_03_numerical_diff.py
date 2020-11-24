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


# %% numerical gradient
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


# %%

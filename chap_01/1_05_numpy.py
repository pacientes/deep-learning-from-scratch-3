# %% import numpy
import numpy as np

# %% build numpy array
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))

# %% numpy numeric operator
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)

print(x - y)

# 원소별 곱셈, element-wise product
print(x * y)

# 원소별 나눗셈, element-wise division
print(x / y)

# %% sample broadcast
x = np.array([1.0, 2.0, 3.0])
print(x / 2.0)

# %% n-dimension numpy array
A = np.array([[1, 2], [3, 4]])
print(A)

print(A.shape)

print(A.dtype)

# %% array operators
B = np.array([[3, 0], [0, 6]])
# element-wise add
print(A + B)

# element-wise product
print(A * B)

# %% array broadcast
print(A)
print(A * 10)

# %%

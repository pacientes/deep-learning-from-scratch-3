# %% one-dimension array
import numpy as np
A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A))

print(A.shape)

print(A.shape[0])

# %% two dimension array
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)

print(np.ndim(B))

print(B.shape)

# %% 2-d matrix product
A = np.array([[1, 2], [3, 4]])
print(A.shape)

B = np.array([[5, 6], [7, 8]])
print(B)

print(np.dot(A, B))
# %% other 2-d matrix product
A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))

# %% example... matrix shape is mismatch
C = np.array([[1, 2], [3, 4]])
print(C.shape)

print(A.shape)

print(np.dot(A, C))

# %% Example... matrix product (2-d and 1-d)
A = np.array([[1, 2,], [3, 4], [5, 6]])
print(A.shape)

B = np.array([7, 8])
print(B.shape)

print(np.dot(A, B))

# %% matrix product in neural network
X = np.array([1, 2])
print(X.shape)

W = np.array([[1, 3, 5], [2, 4, 6]])
print(W)
print(W.shape)

Y = np.dot(X, W)
print(Y)

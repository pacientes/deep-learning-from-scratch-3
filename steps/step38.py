if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import dezero.functions as F
from dezero import Variable

# x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
# y = F.reshape(x, (6,))
# y.backward(retain_grad=True)

# print(x.grad)


# Variable reshape 테스트

# x = Variable(np.random.randn(1, 2, 3))
# print(x)
# y = x.reshape((2, 3))
# print(y)
# y = x.reshape(2, 3)
# print(y)


# Transpose 테스트

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x)
y = F.transpose(x)
y.backward()

print(x.grad)
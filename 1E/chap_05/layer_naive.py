class MulLayer:
    def __init__(self) -> None:
        self.x = None
        self.y = None

    def forward(self, x, y) -> float:
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout) -> float:
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self) -> None:
        pass

    def forward(self, x, y) -> float:
        out = x + y

        return out

    def backward(self, dout) -> float:
        dx = dout * 1
        dy = dout * 1

        return dx, dy


class Relu:
    def __init__(self) -> None:
        self.mask = None

    def forward(self, x):
        self.mask = x <= 0
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx


import numpy as np


class Sigmoid:
    def __init__(self) -> None:
        super().__init__()
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out

        return dx
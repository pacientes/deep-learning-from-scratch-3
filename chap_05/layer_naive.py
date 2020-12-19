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

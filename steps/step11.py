import numpy as np
import unittest


class Variable:
    def __init__(self, data) -> None:
        # 데이터 입력 타입 강제화
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}는 지원하지 않습니다.")

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    # 계산 효율을 위해 재귀 -> 반복문 방식으로 고친다.
    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]

        while funcs:
            f = funcs.pop()  # 함수를 가져온다.
            x, y = f.input, f.output  # 함수의 입력과 출력을 가져온다.
            x.grad = f.backward(y.grad)  # backward 메서드를 호출한다.

            if x.creator is not None:
                funcs.append(x.creator)  # 하나 앞의 함수를 리스트에 추가한다.


# 타입 변환 편의함수
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(xs)
        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs  # 출력 저장
        return outputs

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Square(Function):
    def forward(self, xs):
        return xs ** 2

    def backward(self, gys):
        x = self.input.data
        gx = 2 * x * gys
        return gx


# 클래스 생성 편의함수
def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, xs):
        return np.exp(xs)

    def backward(self, gys):
        x = self.input.data
        gx = np.exp(x) * gys
        return gx


# 클래스 생성 편의함수
def exp(x):
    return Exp()(x)


class Add(Function):
    def forward(self, xs):
        x0, x1 = xs
        y = x0 + x1
        return (y,)


xs = [Variable(np.array(2)), Variable(np.array(3))]
f = Add()
ys = f(xs)
y = ys[0]

print(y.data)
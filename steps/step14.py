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
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    funcs.append(x.creator)

    def cleargrad(self):
        self.grad = None


# 타입 변환 편의함수
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    # 가변 길이 인수 전달
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)

        outputs = [Variable(as_array(y)) for y in ys]
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs  # 출력 저장
        # outputs에 원소가 하나라면, 리스트가 아닌 원소 자체를 반환
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gys):
        raise NotImplementedError


class Square(Function):
    def forward(self, xs):
        return xs ** 2

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
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
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    def backward(self, gy):
        return gy, gy


# 클래스 생성 편의함수
def add(x0, x1):
    return Add()(x0, x1)


x = Variable(np.array(3.0))

y = add(x, x)
y.backward()

print(x.grad)


x.cleargrad()  # 두 번째 계산 시 값 초기화
y = add(add(x, x), x)
y.backward()

print(x.grad)
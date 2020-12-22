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
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y))
        output.set_creator(self)
        self.input = input
        self.output = output  # 출력 저장
        return output

    def forward(self, x):
        raise NotImplementedError

    def backward(self, gy):
        raise NotImplementedError


class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    def forward(self, x):
        return np.exp(x)

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# 클래스 생성 편의함수
def square(x):
    return Square()(x)


# 클래스 생성 편의함수
def exp(x):
    return Exp()(x)


# 유닛 테스트 도구
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)


class SquareTest(unittest.TestCase):
    def test_forward(self):
        x = Variable(np.array(2.0))
        y = square(x)
        expected = np.array(4.0)
        self.assertEqual(y.data, expected)

    def test_backward(self):
        x = Variable(np.array(3.0))
        y = square(x)
        y.backward()
        expected = np.array(6.0)
        self.assertEqual(x.grad, expected)

    def test_gradient_check(self):
        x = Variable(np.random.rand(1))
        y = square(x)
        y.backward()
        num_grad = numerical_diff(square, x)
        flg = np.allclose(x.grad, num_grad)
        self.assertTrue(flg)


x = Variable(np.array(0.5))

y = square(exp(square(x)))
y.backward()
print(f"자동 계산한 역전파 {x.grad}")
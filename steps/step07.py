import numpy as np


class Variable:
    def __init__(self, data) -> None:
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        f = self.creator  # 1. 함수를 가져온다.

        if f is not None:
            x = f.input  # 2. 함수의 입력을 가져온다.
            x.grad = f.backward(self.grad)  # 3. 함수의 backward 메서드를 호출한다.
            x.backward()  # 하나 앞 변수의 backward를 호출한다.(재귀)


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
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


A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

# forward 계산
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라간다.
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x


# 역전파 도전 - 수동
y.grad = np.array(1.0)

C = y.creator  # 1. 함수를 가져온다.
b = C.input  # 2. 함수의 입력을 가져온다.
b.grad = C.backward(y.grad)  # 3. 함수의 backward 메서드를 호출한다.

B = b.creator  # 1. 함수를 가져온다.
a = B.input  # 2. 함수의 입력을 가져온다.
a.grad = B.backward(b.grad)  # 3. 함수의 backward 메서드를 호출한다.

A = a.creator  # 1. 함수를 가져온다.
x = A.input  # 2. 함수의 입력을 가져온다.
x.grad = A.backward(a.grad)  # 3. 함수의 backward 메서드를 호출한다.

print(f"수동 계산한 역전파 {x.grad}")

# 역전파 도전 - 자동

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))

# forward 계산
a = A(x)
b = B(a)
y = C(b)

y.grad = np.array(1.0)
y.backward()
print(f"자동 계산한 역전파 {x.grad}")
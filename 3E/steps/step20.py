import numpy as np
import unittest  # 테스트 도구
import weakref  # weak reference
import contextlib


class Config:
    enable_backprop = True


@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)


# 역전파 계산 비활성화 헬퍼 함수
def no_grad():
    return using_config("enable_backprop", False)


class Variable:
    def __init__(self, data, name=None) -> None:
        # 데이터 입력 타입 강제화
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError(f"{type(data)}는 지원하지 않습니다.")

        self.data = data
        self.name = name
        self.grad = None
        self.creator = None
        self.generation = 0

    # using len()
    def __len__(self):
        return len(self.data)

    # using print(x)
    def __repr__(self) -> str:
        if self.data is None:
            return "variable(None)"
        p = str(self.data).replace("\n", "\n" + " " * 9)
        return f"variable({p})"

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1  # 세대를 기록한다. (부모 + 1)

    # 계산 효율을 위해 재귀 -> 반복문 방식으로 고친다.
    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()  # 함수를 가져온다.
            gys = [output().grad for output in f.outputs]  # weak reference로 변환
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)

            if not retain_grad:
                for y in f.outputs:
                    y().grad = None  # y는 약한 참조(weak_ref)

    def cleargard(self):
        self.grad = None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def ndim(self):
        return self.data.ndim

    # # operator overload but old method
    # def __mul__(self, other):
    #     return mul(self, other)


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

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]  # weak reference로 변환
            # outputs에 원소가 하나라면, 리스트가 아닌 원소 자체를 반환

        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):
        raise NotImplementedError

    def backward(self, gy):
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

    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
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


class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y

    def backward(self, gy):
        x0, x1 = self.inputs[0].data, self.inputs[1].data
        return gy * x1, gy * x0


def mul(x0, x1):
    return Mul()(x0, x1)


Variable.__mul__ = mul
Variable.__add__ = add


a = Variable(np.array(3.0))
b = Variable(np.array(2.0))
c = Variable(np.array(1.0))

y = a * b + c
y.backward()

print(y)
print(a.grad)
print(b.grad)

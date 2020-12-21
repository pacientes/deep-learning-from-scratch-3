# 1.3 파이썬 인터프리터

# %% operators
print(1 - 2)
print(4 + 5)
print(7 / 5)
print(3 ** 2)


# %% data type
print(type(10))
print(type(2.718))
print(type("hello"))


# %% variable
x = 10
print(x)

x = 100
print(x)

y = 3.14
print(x * y)
print(type(x * y))


# %% list
a = [1, 2, 3, 4, 5]
print(a)

# 리스트 길이
print(len(a))

# 첫 번째 원소
print(a[0])

# 다섯 번째 원소
print(a[4])

# 값 대입
a[4] = 99
print(a)


# %% 리스트 계속
print(a)

# 리스트 슬라이싱
print(a[0:2])  # 인덱스 0부터 2까지 (인덱스 2는 X)
print(a[1:])  # 인덱스 1부터 끝까지
print(a[:3])  # 인덱스 0부터 3까지 (인덱스 3은 X)
print(a[:-1])  # 처음부터 마지막 원소의 1개 앞까지
print(a[:-2])  # 처음부터 마지막 원소의 2개 앞까지


# %% 딕셔너리
me = {"height": 100}
print(me["height"])  # 원소 접근

me["weight"] = 70  # 원소 추가
print(me)


# %% boolean
hungry = True
sleepy = False
print(type(hungry))

# NOT 연산
print(not hungry)

# AND 연산
print(hungry and sleepy)

# OR 연산
print(hungry or sleepy)


# %% if statement
hungry = True
if hungry:
    print("I'm hungry")

hungry = False
if hungry:
    print("I'm hungry")
else:
    print("I'm not hungry")
    print("I'm sleepy")


# %% for loop
for i in [1, 2, 3]:
    print(i)


# %% function
def hello():
    print("Hello World!")


hello()


# %% function continue
def hello(object):
    print("Hello " + object + "!")


hello("cat")

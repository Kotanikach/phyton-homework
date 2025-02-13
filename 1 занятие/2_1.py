import numpy as np
import sys
import array

#Типы данных
x = 1
print(type(x))
print(sys.getsizeof(x))

x = 'hello'
print(type(x))

x = True
print(type(x))

l1 = list([])
print(sys.getsizeof(l1))

l2 = list([1, 2, 3])
print(sys.getsizeof(l2))

l3 = list([1, "2", True])
print(sys.getsizeof(l3))

al = array.array('i', [1, 2, 3]) # все элементы одного типа
print(sys.getsizeof(al))
print(type(al))

a = np.array([1, 2, 3, 4, 5])
print(type(a), a)

#"Повышающее" приведение типов
a = np.array([1.23, 2, 3, 4, 5])
print(type(a), a)

a = np.array([1.23, 2, 3, 4, 5], dtype=int)
print(type(a), a)

a = np.array([range(i, i+3) for i in [2, 4, 6]])
print(a, type(a))

a = np.zeros(10, dtype=int)
print(a, type(a))

print(np.ones((3,5), dtype=float))

print(np.full((4,5), 3.1415))

print(np.arange(0, 20, 2))

print(np.eye(4))

### МАССИВЫ

np.random.seed(1)

x1 = np.random.randint(10, size=3)
x2 = np.random.randint(10, size=(3, 2))
x3 = np.random.randint(10, size=(3, 2, 2))
print(x1, x2, x3, sep="\n")

print(x1.ndim, x1.shape, x1.size)
print(x2.ndim, x2.shape, x2.size)
print(x3.ndim, x3.shape, x3.size)

a = np.array([1, 2, 3, 4, 5])
print(a[0])
print(a[-2])

a[1] = 20
print(a)

a = np.array([[1, 2], [3, 4]])
print(a)

print(a[0, 0])
print(a[-1, -1])

a[1, 0] = 100
print(a)

a = np.array([1, 2, 3, 4, 5])
b = np.array([1.0, 2, 3, 4])

print(a)
print(b)

a[0] = 10
print(a)

a[0] = 10.123
print(a) # останется int

# Срез [s:f:st]

a = np.array([1, 2, 3, 4, 5, 6])

print(a[:3:])
print(a[3:])
print(a[1:-1])
print(a[1::2])
print(a[::-1])

b = a[:3]
print(b)

b[0] =100
print(a)
# срез это не копия, а ссылка

a = np.arange(1, 13)
print(a)
print(a.reshape(2, 6))
print(a.reshape(3, 4))

x = np.array([1, 2, 3])
y = np.array([4, 5])
z = np.array([6])

print(np.concatenate([x,y,z]))

x = np.array([1, 2, 3])
y = np.array([4, 5, 6])

rl = np.vstack([x, y])
print(rl)

print(np.hstack([rl, rl]))

### Вычисления с массивами

# Векторизированная операция - независимо к каждому элементу

x = np.arange(10)
print(x)

print(x*2 + 1)

# Универсальные функции

print(np.add(np.multiply(x, 2), 1))

# - - / // ** %

## np.abs, sin/cos/tan, exp, log

x = np.arange(5)

y = np.zeros(10)
print(np.multiply(x, 10, out=y[::2]))

print(y)

x = np.arange(1, 5)
print(x)

print(np.add.reduce(x))
print(np.add.accumulate(x))

x = np.arange(1, 10)
print(np.multiply.outer(x, x))

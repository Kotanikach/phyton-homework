import numpy as np
import pandas as pd

# Pandas - расширение NumPy (структурированные массивы). Строки и столбцы индексируются метками, а не только числовыми значениями

# Series, DataFrame, Index

## Series
data = pd.Series([0.25, 0.5, 0.75, 1.0])
print(data)
print(type(data))

print(data.values)
print(type(data.values))
print(data.index)
print(type(data.index))

print(data[0])
print(data[1:3])

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])
print(data)
print(data['a'])
print(data['b':'d'])

print(type(data.index))

data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 10, 'c', 'd']) # тип для индексов можно использовать разный
print(data)

print(data[1])
print(data[10:'d'])

population_dict = {
    'city1': 1001,
    'city2': 1002,
    'city3': 1003,
    'city4': 1004,
    'city5': 1005
}
population = pd.Series(population_dict)
print(population)
print(population['city4'])
print(population['city4':'city5'])

# Для создания Series можно использовать:
# - списки Python или массив NumPy
# - скалярные значения
# - словари

## DataFrame - двумерный массив с явно определенными индексами. Последовательность согласованных объектов Series
population_dict = {
    'city1': 1001,
    'city2': 1002,
    'city3': 1003,
    'city4': 1004,
    'city5': 1005
}

area_dict = {
    'city1': 9991,
    'city2': 9992,
    'city3': 9993,
    'city4': 9994,
    'city5': 9995
}

population = pd.Series(population_dict)
area = pd.Series(area_dict)
print(population)
print(area)

states = pd.DataFrame({
    'population': population,
    'area': area
})

print(states)
print(states.values)
print(states.index)
print(states.columns)

print(type(states.values))
print(type(states.index))
print(type(states.columns))

print(states['area'])

# DataFrame. Способы создания
# - через объекты Series
# - списки словарей
# - словари объектов Series
# - двумерный массив NumPy
# - структурированный массив Numpy

## Index - способ организации ссылки на данные объектов Series и DataFrame. Index - неизменяем и упорядочен мультимножество (могут быть повторяющиеся элементы)
ind = pd.Index([2, 3, 5, 7, 11])
print(ind[1])
print(ind[::2])

# Index - следует соглашениям объекта set
indA = pd.Index([1, 2, 3, 4, 5])
indB = pd.Index([2, 3, 4, 5, 6])
print(indA.intersection(indB))

# Выборка данных Series
## Как словарь
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

print('a' in data)
print('z' in data)

print(data.keys())
print(list(data.items()))

data['a'] = 100
data['z'] = 1000
print(data)

## Как одномерный массив
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=['a', 'b', 'c', 'd'])

print(data['a':'c'])
print(data[0:2])
print(data[(data > 0.5) & (data < 1)])
print(data[['a', 'd']])

# атрибуты-индексаторы
data = pd.Series([0.25, 0.5, 0.75, 1.0], index=[1, 3, 10, 15])

print(data[1]) # 0.25
print(data.loc[1]) # 0.25
print(data.iloc[1]) # 0.5

# Выбор данных из DataFrame
# как словарь
pop = pd.Series(
    {
        'city1': 1001,
        'city2': 1002,
        'city3': 1003,
        'city4': 1004,
        'city5': 1005
    }
)

area = pd.Series(
    {
        'city1': 9991,
        'city2': 9992,
        'city3': 9993,
        'city4': 9994,
        'city5': 9995
    }
)

data = pd.DataFrame({'areal':area, 'popl': pop, 'pop': pop})
print(data)

print(data['areal'])
print(data.areal)

print(data.popl is data['popl'])
print(data.pop is data['pop'])

data['new'] = data['areal']
data['newl'] = data['areal'] / data['popl']
print(data)

# двумерный NumPy-массив
data = pd.DataFrame({'areal':area, 'popl': pop})

print(data)
print(data.values)
print(data.T)
print(data['areal'])
print(data.values[0])

# атрибуты-индексаторы
data = pd.DataFrame({'areal':area, 'popl': pop, 'pop': pop})
print(data)
print(data.iloc[:3, 1:2])
print(data.loc[:'city4', 'popl':'pop'])
print(data.loc[data['pop'] > 1002, ['areal','pop']])

data.iloc[0, 2] = 999999
print(data)


rng = np.random.default_rng()
s = pd.Series(rng.integers(0, 10, 4))

print(s)
print(np.exp(s))

pop = pd.Series(
    {
        'city1': 1001,
        'city2': 1002,
        'city3': 1003,
        'city41': 1004,
        'city51': 1005
    }
)

area = pd.Series(
    {
        'city1': 9991,
        'city2': 9992,
        'city3': 9993,
        'city42': 9994,
        'city52': 9995
    }
)

data = pd.DataFrame({'areal': area, 'popl': pop})
print(data)
#NaN - not a number

dfA = pd.DataFrame(rng.integers(0, 10, (2, 2)), columns=['a', 'b'])
dfB = pd.DataFrame(rng.integers(0, 10, (3, 3)), columns=['a', 'b', 'c'])
print(dfA, dfB, sep='\n')

print(dfA + dfB)

rng = np.random.default_rng(1)
A = rng.integers(1, 10, (3, 4))
print(A)
print(A[0])
print(A - A[0])

df = pd.DataFrame(A, columns=['a', 'b', 'c', 'd'])
print(df)
print(df.iloc[0])
print(df - df.iloc[0])

print(df.iloc[0, ::2])
print(df - df.iloc[0, ::2])

# NA-значения: NaN, null, -99999

# Pandas. Два способа хранения отсутствующих значений
# индикаторы Nan, None
# null

# None - объект (накладные расходы). Не работает с sum, min

# val1 = np.array([1, None, 2, 3])
# print(val1.sum()) # не работает

val1 = np.array([1, np.nan, 2, 3])
print(val1)
print(val1.sum()) # nan
print(np.sum(val1)) # nan
print(np.nansum(val1)) # 6

x = pd.Series(range(10), dtype=int)
print(x)

x[0] = None
x[1] = np.nan
print(x)

x1 = pd.Series(['a', 'b', 'c'])
x1[0] = None
x1[1] = np.nan
print(x1)

x2 = pd.Series([1, 2, 3, np.nan, None, pd.NA])
print(x2)

x3 = pd.Series([1, 2, 3, np.nan, None, pd.NA], dtype='Int32')
print(x3)

print(x3.isnull())
print(x3[x3.isnull()])
print(x3[x3.notnull()])
print(x3.dropna())

df = pd.DataFrame(
    [
        [1, 2, 3, np.nan, None, pd.NA],
        [1, 2, 3, None, 5, 6],
        [1, np.nan, 3, None, np.nan, 6],
])

print(df)
print(df.dropna())
print('\n')
print(df.dropna(axis=0)) # строки
print(df.dropna(axis=1)) # столбцы

# how:
# - all - все значения NA
# - any - хотя бы одно значение
# - thresh = x, остается, если присутствует минимум х не пустых значений
print(df.dropna(axis=1, how='all')) # отбросятся столбцы, где весь столбец NA, а в которых не все NA останется
print(df.dropna(axis=1, how='any'))
print(df.dropna(axis=1, thresh=2))

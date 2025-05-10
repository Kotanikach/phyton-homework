# Переобучение и дисперсия

# Цель не в минимизации сумм квадратов до точек, а в том, чтобы делать правильные предсказания на новых данных

# Переобученные модели - чувствительны к выбросам (сильным отклонениям)
# в прогнозах высокая дисперсия, поэтому к моделям специально добавляется смещение

# Смещение модели - при постройке предпочтение отдается определенной схеме (прямая линия или к примеру должна пройти через (0, 0))
#и такая структура должна минимизировать остатки

# Если в модель добавить смещение - есть риск недообучить

# Задача сводится к балансировке: минимизировать функцию потерь + не переобучиться

# Гребневая регрессия добавляется смещение в виде некоторого штрафа => из-за этого хуже идет подгонка под данные

# Лассо-регрессия - удаление некоторых переменных => снижается размерность

# Механически применить линейную регрессию к данным - сделать прогноз, и думать, что все ок нельзя
# Даже в случае прямой линии, модель может быть переобучена

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_val_score

data = np.array(
    [
        [1, 5],
        [2, 7],
        [3, 7],
        [4, 10],
        [5, 11],
        [6, 14],
        [7, 17],
        [8, 19],
        [9, 22],
        [10, 28]
    ]
)

# Градиентный спуск - пакетный градиентный спуск. Для работы используются все обучающие данные
# на практике используется его разновидность - стохастический град спуск - на каждой итерации обучаемся только по одной выборке из данных
# - сокращение числа вычислений
# - вносим смещение -> боремся с переобучением
# мини-пакеты градиентный спуск, на каждой итерации используется несколько выборок

x = data[:, 0]
y = data[:, 1]

n = len(x)
w1 = 0.0
w0 = 0.0
L = 0.001
sample_size = 2  # размер выборки
iterations = 100_000

for i in range(iterations):
    idx = np.random.choice(n, sample_size, replace=False)

    D_w0 = 2 * sum(-y[idx] + w0 + w1 * x[idx] )
    D_w1 = 2 * sum(x[idx] * (-y[idx] + w0 + w1 * x[idx]) )
    w1 -= L * D_w1
    w0 -= L * D_w0

print(w1, w0)

# как оценить, насколько "промахиваются" прогнозы при использовании лин регр
# Для оценки сп

# data_df = pd.DataFrame(data)
# print(data_df.corr(method = "pearson"))

# data_df[1] = data_df[1].values[::-1]
# print(data_df.corr(method = "pearson"))

# коэффициент корреляции позволяет понять, есть ли связь между двумя переменными

# Обучающие и тестовые выборки - основной метод борьбы с переобучением
# Набор данных делится на обучающую и тестовую выборки

# Во всех видах машинного обучения с учителем это встречается
# Обычная пропорция - 2/3 обучение, 1/3 на тест. (еще бывают (4 к 5) (9 к 10)...)

data_df = pd.DataFrame(data)

X = data_df.values[:,:-1]
Y = data_df.values[:, -1]

# print(type(X))
print(X)
print(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3)
print(X_train)
print(Y_train)

print(X_test)
print(Y_test)

# model = LinearRegression()
# model.fit(X_train, Y_train)
# r = model.score(X_test, Y_test) # Коэффициент детерминации
# print(r)

kfold = KFold(n_splits=3, random_state=1, shuffle=True) # 3-x кратная перекрестная валидация
model = LinearRegression()
results = cross_val_score(model, X, Y, cv=kfold)

print(results) # средне-квадратические ошибки
print(results.mean(), results.std())

# метрики показывают, как ЕДИНООБРАЗНО ведет себя модель на разных выборках
# Возможно использование поэлементной перекрестной валидации (если данных мало)
# случайную валидацию делать, когда большой разброс

# Валидационная выборка - для сравнения различных моделей или конфигураций


# Многомерная линейная регрессия

data_df = pd.read_csv("./multiple_independent_variable_linear.csv")
# print(data_df.head())

X = data_df.values[:, :-1]
Y = data_df.values[:, -1]
model = LinearRegression().fit(X,Y)

print(model.coef_, model.intercept_)

x1 = X[:, 0]
x2= X[:, 1]

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, Y)


x1_ = np.linspace(min(x1), max(x1), 100)
x2_ = np.linspace(min(x2), max(x2), 100)

X1_, X2_ = np.meshgrid(x1_, x2_)
Y_ = model.intercept_ + model.coef_[0] * X1_ + model.coef_[1] * X2_
ax.plot_surface(X1_, X2_, Y_, alpha = 0.4) #, rstride=1, cstride=1)
plt.show()
# Наивная байесовская классификация
# набор моделей, которые предлагают быстрые и простые алгоритмы классификации
# Теорема/формула Байеса (формула для апостериорной вероятности)
# Наивно допущение относительно генеративной модели
# получаем грубое приближение для каждого класса
# Чаще всего используют гауссовские упущения

# Гауссовский наивный байесовский классификатор
# Наивное допущение состоит в том, что ! данные всех категорий взяты из простого нормального распределения !

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

iris = sns.load_dataset("iris")
# print(iris.head())

# sns.pairplot(iris, hue = "species")
# plt.show()

data = iris[["sepal_length", "petal_length", "species"]]

print(data.shape)

# setosa versicolor
# setosa virginica

data_df = data[ (data["species"] == "setosa") | (data["species"] == "versicolor")]
print(data_df.shape)


X = data_df[["sepal_length", "petal_length"]]
Y = data_df[["species"]]

print(type(X))

model = GaussianNB()
model.fit(X, Y)

print(model.theta_[0]) # мат ожидание
print(model.theta_[1])
print(model.var_[0]) # дисперсия
print(model.var_[1])


data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"], data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T, # ravel преобразует двумерный в одномерный
    columns=["sepal_length", "petal_length"],
)
print(X_p.head())

Y_p = model.predict(X_p)

X_p["species"] = Y_p
X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]
print(X_p.head())

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)


theta0 = model.theta_[0]
theta1 = model.theta_[1]
var0 = model.var_[0]
var1 = model.var_[1]

# формула нормального распределения
z1 = (1/(1 * np.pi * (var0[0] * var0[1]) ** 0.5) *
      np.exp(-0.5 * (X1_p - theta0[0]) ** 2/var0[0] +
             -0.5 * (X2_p - theta0[1]) ** 2/var0[1]))

z2 = (1/(1 * np.pi * (var1[0] * var1[1]) ** 0.5) *
      np.exp(-0.5 * (X1_p - theta1[0]) ** 2/var1[0] +
             -0.5 * (X2_p - theta1[1]) ** 2/var1[1]))

plt.contour(X1_p, X2_p, z1)
plt.contour(X1_p, X2_p, z2)
plt.show()

fig = plt.figure()
ax = plt.axes(projection = "3d")
ax.contour3D(X1_p, X2_p, z1, 40)
ax.contour3D(X1_p, X2_p, z2, 40)

plt.show()

# virginica
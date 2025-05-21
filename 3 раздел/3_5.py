# Метод опорных векторов (SVM - support vector machine) - классификация и регрессия
# Разделяющая классификация
# выбирается линия с максимальным отступом

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[ (data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
Y = data_df[["species"]]

data_df_seposa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_seposa["sepal_length"], data_df_seposa["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

# plt.show()


# Если С большое, то отступ задается жестко. Чем меньше С, тем отступ становится более размытым

model = SVC(kernel="linear", C=10000)
model.fit(X, Y)

print(model.support_vectors_)

plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
            s = 400,
            facecolors = "none",
            edgecolor = "k")


x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=["sepal_length", "petal_length"],
)

Y_p = model.predict(X_p)
X_p["species"] = Y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

plt.scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.4)
plt.scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.4)

# plt.show()

# В случае, если данные перекрываются, то идеальной границы не существует - у модели существует гиперпараметр, который определяет "размытие" отступа

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[ (data["species"] == "virginica") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
Y = data_df[["species"]]

print(X)

data_df_virginica = data_df[data_df["species"] == "virginica"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

plt.scatter(data_df_virginica["sepal_length"], data_df_virginica["petal_length"])
plt.scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

# plt.show()


c_value = [[10000, 1000, 100, 10], [1, 0.1, 0.01, 0.001]]

fig, ax = plt.subplots(2, 4, sharex="col", sharey="row")

for i in range(2):
    for j in range(4):
        ax[i, j].scatter(data_df_virginica["sepal_length"], data_df_virginica["petal_length"])
        ax[i, j].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

        model = SVC(kernel="linear", C=c_value[i][j])
        model.fit(X, Y)

        ax[i, j].scatter(model.support_vectors_[:, 0],
                         model.support_vectors_[:, 1],
                            s=100,
                            facecolors="none",
                            edgecolor="k")

        x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
        x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

        X1_p, X2_p = np.meshgrid(x1_p, x2_p)

        X_p = pd.DataFrame(
            np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
            columns=["sepal_length", "petal_length"],
        )

        Y_p = model.predict(X_p)
        X_p["species"] = Y_p

        X_p_virginica = X_p[X_p["species"] == "virginica"]
        X_p_versicolor = X_p[X_p["species"] == "versicolor"]

        ax[i, j].scatter(X_p_virginica["sepal_length"], X_p_virginica["petal_length"], alpha=0.01)
        ax[i, j].scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.01 )

plt.show()

# Достоинства метода:
# Модель зависит от небольшого числа опорных векторов => компактность модели
# После обучения предсказания проходят очень быстро
# На работу метода влияют только точки, находящиеся около отступов, поэтому эти методы подходят для многомерных данных


# Недостатки:
# При большом количестве обучающих образцов могут быть вычислительные затраты
# большая зависимость от размытости ( С ) Поиск этого параметра может привести к большим вычислительным затратам
# У результатов отсутствует вероятностная интерпретация
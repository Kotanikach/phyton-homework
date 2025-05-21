import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.svm import SVC

fig, ax = plt.subplots(1, 2, figsize=(10, 5))

iris = sns.load_dataset("iris")

data = iris[["sepal_length", "petal_length", "species"]]
data_df = data[(data["species"] == "setosa") | (data["species"] == "versicolor")]

X = data_df[["sepal_length", "petal_length"]]
Y = data_df[["species"]]

data_df_setosa = data_df[data_df["species"] == "setosa"]
data_df_versicolor = data_df[data_df["species"] == "versicolor"]

ax[0].scatter(data_df_setosa["sepal_length"], data_df_setosa["petal_length"])
ax[0].scatter(data_df_versicolor["sepal_length"], data_df_versicolor["petal_length"])

model1 = SVC(kernel="linear", C=10000)
model1.fit(X, Y)

ax[0].scatter(model1.support_vectors_[:, 0], model1.support_vectors_[:, 1],
              s=400,
              facecolors="none",
              edgecolor="k")

x1_p = np.linspace(min(data_df["sepal_length"]), max(data_df["sepal_length"]), 100)
x2_p = np.linspace(min(data_df["petal_length"]), max(data_df["petal_length"]), 100)

X1_p, X2_p = np.meshgrid(x1_p, x2_p)

X_p = pd.DataFrame(
    np.vstack([X1_p.ravel(), X2_p.ravel()]).T,
    columns=["sepal_length", "petal_length"],
)

Y_p = model1.predict(X_p)
X_p["species"] = Y_p

X_p_setosa = X_p[X_p["species"] == "setosa"]
X_p_versicolor = X_p[X_p["species"] == "versicolor"]

ax[0].scatter(X_p_setosa["sepal_length"], X_p_setosa["petal_length"], alpha=0.1)
ax[0].scatter(X_p_versicolor["sepal_length"], X_p_versicolor["petal_length"], alpha=0.1)
# ------------------------------------------------------------
def check_support_vectors(values, vectors):
    for i in vectors:
        if (values[0] == i[0] and values[1] == i[1]):
            return True
    return False

data_df_cropped = data_df.copy()
del_indexes = []
for i in range(1, data_df_cropped.shape[0]):
    if (not check_support_vectors(data_df_cropped.values[i, :2], model1.support_vectors_) and i % 2):
        del_indexes.append(i)
data_df_cropped = data_df_cropped.drop(del_indexes)
# Убрали половину точек, которые не являются опорными

X_cropped = data_df_cropped[["sepal_length", "petal_length"]]
Y_cropped = data_df_cropped[["species"]]

data_df_setosa_cropped = data_df_cropped[data_df_cropped["species"] == "setosa"]
data_df_versicolor_cropped = data_df_cropped[data_df_cropped["species"] == "versicolor"]

ax[1].scatter(data_df_setosa_cropped["sepal_length"], data_df_setosa_cropped["petal_length"])
ax[1].scatter(data_df_versicolor_cropped["sepal_length"], data_df_versicolor_cropped["petal_length"])

model2 = SVC(kernel="linear", C=10000)
model2.fit(X_cropped, Y_cropped)

ax[1].scatter(model2.support_vectors_[:, 0], model2.support_vectors_[:, 1],
              s=400,
              facecolors="none",
              edgecolor="k")

x1_p = np.linspace(min(data_df_cropped["sepal_length"]), max(data_df_cropped["sepal_length"]), 100)
x2_p = np.linspace(min(data_df_cropped["petal_length"]), max(data_df_cropped["petal_length"]), 100)

X1_p_cropped, X2_p_cropped = np.meshgrid(x1_p, x2_p)

X_p_cropped = pd.DataFrame(
    np.vstack([X1_p_cropped.ravel(), X2_p_cropped.ravel()]).T,
    columns=["sepal_length", "petal_length"],
)

Y_p_cropped = model1.predict(X_p_cropped)
X_p_cropped["species"] = Y_p_cropped

X_p_setosa_cropped = X_p_cropped[X_p_cropped["species"] == "setosa"]
X_p_versicolor_cropped = X_p_cropped[X_p_cropped["species"] == "versicolor"]

ax[1].scatter(X_p_setosa_cropped["sepal_length"], X_p_setosa_cropped["petal_length"], alpha=0.1)
ax[1].scatter(X_p_versicolor_cropped["sepal_length"], X_p_versicolor_cropped["petal_length"], alpha=0.1)

ax[0].set(title="All points")
ax[1].set(title="Half of the points")

plt.savefig("3_5_homework.png")

plt.show()
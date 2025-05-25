import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Обучение с учителем (классификация)

df = px.data.iris()

# два сорта
species_to_use = ["setosa", "versicolor"]
df_subset = df[df["species"].isin(species_to_use)]

# 1. Метод опорных векторов (SVM)
X = df_subset[["sepal_length", "sepal_width"]].values
y = df_subset["species"].map({"setosa": 0, "versicolor": 1}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SVC(kernel='linear')
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"SVM Accuracy: {accuracy}")

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()])

Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("SVM")
# plt.show()
plt.savefig("3_7_homework_1_SVM.png")

# 2. Метод главных компонент (PCA)
X = df_subset[["sepal_length", "sepal_width", "petal_length", "petal_width"]].values
y = df_subset["species"].map({"setosa": 0, "versicolor": 1}).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_train)

model = SVC(kernel='linear')
model.fit(X_pca, y_train)

X_test_pca = pca.transform(X_test)

accuracy = model.score(X_test_pca, y_test)
print(f"PCA + SVM Accuracy: {accuracy}")

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, cmap=plt.cm.Paired, edgecolors='k')

ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))

Z = model.decision_function(pca.transform(np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.ravel().shape), np.zeros(xx.ravel().shape)])))

Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[0], linestyles=['-'])


plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("PCA with SVM")
# plt.show()
plt.savefig("3_7_homework_1_PCA.png")

# Обучение без учителя (классификация)

# 3. Метод k-средних (K-Means)
X = df_subset[["sepal_length", "sepal_width"]].values

kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
kmeans.fit(X)
labels = kmeans.labels_

plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap=plt.cm.Paired, edgecolors='k')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, marker='X', c='red', label='Centroids')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("K-Means")
plt.legend()
# plt.show()
plt.savefig("3_7_homework_1_k-means.png")
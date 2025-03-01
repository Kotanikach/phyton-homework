import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.core.pylabtools import figsize

# 1
x = np.array([1, 5, 10, 15, 20])
y1 = np.array([1, 7, 3, 5, 11])
y2 = np.array([4, 3, 1, 8, 12])

plt.plot(x, y1, 'r-o', label='line 1')
plt.plot(x, y2, 'g-.o', label='line 2')

plt.legend()
# plt.show()


# 2
x = np.arange(1, 6)
y1 = np.array([1, 7, 6, 4, 5])
y2 = np.array([9, 4, 2, 4, 9])
y3 = np.array([-8, -4, 2, -4, -8])

fig = plt.figure()
gs = plt.GridSpec(2, 2)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])

ax1.plot(x, y1)
ax2.plot(x, y2)
ax3.plot(x, y3)

plt.show()


# 3
fig, ax = plt.subplots()
x = np.linspace(-5, 5)
ax.plot(x, x**2)
ax.annotate('min', xy=(0, 0), xytext=(0,10), arrowprops=dict(facecolor='green'))

plt.show()


4
x = np.linspace(0,7,10)
y = np.sin(x) + np.cos(x[:,np.newaxis])
plt.imshow(y, cmap='viridis')
plt.colorbar()

plt.show()


# 5
x=np.linspace(0, 5, 100)
y=np.cos(x*np.pi)
plt.plot(x,y, color='red')
plt.fill_between(x, y)
plt.show()


# 6
x=np.linspace(0, 5, 100)
y=np.cos(x*np.pi)
y_m = np.ma.masked_where(y < -0.5, y)
plt.ylim(-1, 1)
plt.plot(x,y_m)

plt.show()


# 7
x = np.arange(0, 7)
where_set = ['pre', 'post', 'mid']
fig, ax = plt.subplots(1, 3)

for i, axi in enumerate(ax):
  axi.step(x, x, "g-o", where=where_set[i])
  axi.grid()

plt.show()


# 8
x = np.arange(0, 11, 1)

y1 = np.array([(-0.2)*i**2+2*i for i in x])
y2 = np.array([(-0.4)*i**2+4*i for i in x])
y3 = np.array([2*i for i in x])

labels = ["y1", "y2", "y3"]

fig, ax = plt.subplots()
ax.stackplot(x, y1, y2, y3, labels=labels)
ax.legend(loc='upper left')

plt.show()


# 9
vals = [24, 17, 53, 21, 35]
labels = ["Ford", "Toyota", "BMV", "AUDI", "Jaguar"]

fig, ax = plt.subplots()
ax.pie(vals, labels=labels, explode=(0, 0, 0.15, 0, 0))

plt.show()


# 10
fig, ax = plt.subplots()
ax.pie(vals, labels=labels, wedgeprops=dict(width=0.5))

plt.show()
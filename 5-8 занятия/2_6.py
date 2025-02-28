import numpy as np
import matplotlib as mpl
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(1)
data = rng.normal(size=1000)

# bins - количество разных вершин
# histtype - контур
plt.hist(data, bins=30, density=True, alpha=0.5, histtype='step', edgecolor='red') # гистограмма

plt.show()


x1=rng.normal(0,0.8,1000)
x2=rng.normal(-2,1,1000)
x3=rng.normal(3,2,1000)
args=dict(
    alpha=0.3,
    bins=40
)

plt.hist(x1, **args)
plt.hist(x2, **args)
plt.hist(x3, **args)

plt.show()

# показывает какие значения в корзинах и какие значения на границах
print(np.histogram(x1,bins=1))
print(np.histogram(x1,bins=2))
print(np.histogram(x1,bins=40))


# двумерные гистограммы
mean=[0,0]
cov=[[1, 1], [1, 2]]
x, y =rng.multivariate_normal(mean, cov, 10000).T

plt.hist2d(x,y, bins=30)
plt.hexbin(x, y, gridsize=30) # то же как и выше, только шестиугольники
cb = plt.colorbar()
cb.set_label('Point in interval')

plt.show()

print(np.histogram2d(x, y, bins=1))
print(np.histogram2d(x, y, bins=10))


# Легенда
x = np.linspace(0,10,1000)
fig, ax = plt.subplots()

ax.plot(x, np.sin(x), label='Синус')
ax.plot(x, np.cos(x), label='Косинус')
ax.plot(x, np.cos(x) + 2) # в легенде не будет
ax.axis('equal')

# frameon - окантовка легенды
ax.legend(frameon=False, fancybox=True, shadow=True)

plt.show()

x = np.linspace(0,10,1000)
fig, ax = plt.subplots()
y = np.sin(x[:,np.newaxis] + np.pi * np.arange(0,2,0.5))

lines = plt.plot(x,y) # для каждой линии вернет экземпляр plt.line2d

plt.legend(lines, ['1', 'второй', 'third', '4-ый'], loc='upper center')
#plt.legend(lines[:2], ['1', 'второй'])

plt.show()




cities = pd.read_csv('./data/california_cities.csv')
lat, lon, pop, area = cities['latd'], cities['longd'], cities['population_total'], \
    cities['area_total_km2']

plt.scatter(lon, lat, c=np.log10(pop), s=area)
plt.xlabel('Широта')
plt.ylabel('Долгота')
plt.colorbar()
plt.clim(3,7) # установить пределы значений, которые будут отображаться

# plt.legend() # labelов нет - ничего не выведется, только место появится

plt.scatter([],[], c='k', alpha=0.5, s=100, label='100 $km^2$')
plt.scatter([],[], c='k', alpha=0.5, s=300, label='300 $km^2$')
plt.scatter([],[], c='k', alpha=0.5, s=500, label='500 km^2')

plt.legend(labelspacing=3, frameon=False)

plt.show()


fig, ax = plt.subplots()
lines=[]
styles = ['-', '--', '-.', ':']
x = np.linspace(0,10, 1000)
for i in range(4):
    lines+=ax.plot(
        x,
        np.sin(x - i + np.pi/2),
        styles[i]
    )
ax.axis('equal')
ax.legend(lines[:2],['line 1', 'line 2'], loc='upper right')
# создать еще просто одну легенду не можем, надо слоем дополнительным
leg = mpl.legend.Legend(ax,lines[1:], ['line 2', 'line 3', ' line 4'], loc='lower left')
ax.add_artist(leg)

plt.show()


# Шкалы
x= np.linspace(0,10,1000)
y= np.sin(x)+np.cos(x[:,np.newaxis])
plt.imshow(y, cmap='Blues')
plt.colorbar()
plt.show()
# Карты цветов
# - последовательные
# - дивергентные (два цвета)
# - качественные (смешиваются без четкого порядка)

# 1
plt.imshow(y, cmap='binary')
plt.imshow(y, cmap='viridis')

# 2
plt.imshow(y, cmap='RdBu')
plt.imshow(y, cmap='PuOr')

# 3
plt.imshow(y, cmap='rainbow')
plt.imshow(y, cmap='jet')

plt.show()


x= np.linspace(0,10,1000)
y= np.sin(x)+np.cos(x[:,np.newaxis])

plt.figure()
plt.subplot(1,2 ,1)
plt.imshow(y,cmap='viridis')
plt.colorbar()

plt.subplot(1,2 ,2)
# plt.imshow(y,cmap='viridis')
plt.imshow(y,cmap=plt.cm.get_cmap('viridis',6)) # будет дискретное деление
plt.colorbar()
plt.clim(-0.25, 0.25)

plt.show()


# матлабовский стиль
ax1 = plt.axes()
# [нижний угол, левый угол, ширина, высота] в десятичных дробях проценты
ax2 = plt.axes([0.4, 0.3, 0.2 ,0.1])

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()

# ооп стиль
fig=plt.figure()
ax1=fig.add_axes([0.1, 0.6, 0.8, 0.4])
ax2=fig.add_axes([0.1, 0.1, 0.8, 0.4])

ax1.plot(np.sin(x))
ax2.plot(np.cos(x))

plt.show()


# Простые сетки
fig=plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1,7):
    ax = fig.add_subplot(2, 3, i)
    ax.plot(np.sin(x+np.pi/4*i))

plt.show()
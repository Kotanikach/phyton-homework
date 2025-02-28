# 1. Сценарий
# 2. командная оболочка IPython
# 3. Jupyter


# 1
# plt.show() - запускается только один раз
# Figure

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import lineStyles

fig = plt.figure()

x= np.linspace(0,10,100)
plt.plot(x,np.sin(x))
plt.show()
plt.plot(x,np.cos(x)) # так не отобразится, надо перенести перед plt.show()


# 2
# %matplotlib
# import matplotlib.pyplot as plt
# plt.plot(...); если поставлена точка с запятой, то консольная информация не выводится
# plot.draw()


# 3
# %matplotlib inline - в блокнот добавляется статическая картинка
# %matplotlib notebook - в блокнот добавляется интерактивные графики


# объявление fig выше в коде
fig.savefig("saved_image.png")

print(fig.canvas.get_supported_filetypes())


# Два способа вывода графиков
# - Matlab - подобный стиль
# - объектно-ориентированный стиль
# - в ОО стиле

x = np.linspace(0,10,100)
plt.figure()

plt.subplot(2, 1, 1) # аналогично в matlabe
plt.plot(x, np.sin(x))

plt.subplot(2, 1, 2)
plt.plot(x, np.cos(x))

# plt.show()
# вернуться к предыдущему графику сложно


fig, ax = plt.subplots(2)

ax[0].plot(x,np.sin(x))
ax[1].plot(x,np.cos(x))
# fig:plt.Figure - контейнер содержит объекты (СК, тексты, метки), ax:Axes - система координат - прямоугольник, деления, метки
# plt.show()


# Цвета линий color
# - 'blue'
# - 'rgbcmyk' -> 'rg
# - '0.14' -> градация серого от 0 до 1
# - RRGGBB - 'FF00EE
# - RGB - (1.0, 0.2, 0.3)
# - HTML - 'salmon'

# Стиль линии
# - сплошная '-', 'solid'
# - штриховая '--', 'dashed'
# - штрих-пунктирная '-.', 'dashdot'
# - пунктирная ':', 'dotted'

# есть маркеры точек, как в матлабе, звездочки, точки пятиугольники и тп

fig = plt.figure()
ax = plt.axes()

ax.plot(x,np.sin(x), color='blue')
ax.plot(x,np.sin(x-1), color='g', linestyle='solid')
ax.plot(x,np.sin(x-2), color='0.75', linestyle='dashed')
ax.plot(x,np.sin(x-3), color='#FF00EE', linestyle='dashdot')
ax.plot(x,np.sin(x-4), color=(1.0, 0.2, 0.3), linestyle='dotted')
ax.plot(x,np.sin(x-5), color='salmon')
ax.plot(x,np.sin(x-6), '--k')

# plt.show()

# форматирование отрезков отображения ох и оу
fix, ax = plt.subplots(4)

ax[0].plot(x, np.sin(x))
ax[1].plot(x, np.sin(x))
ax[2].plot(x, np.sin(x))
ax[3].plot(x, np.sin(x))

ax[1].set_xlim(-2, 12)
ax[1].set_ylim(-1.5, 1.5)

ax[2].set_xlim(12, -2)
ax[2].set_ylim(1.5, -1.5) # отзеркаленный график

ax[3].autoscale(tight=True)

# plt.show()


plt.subplot(3,1,1)

plt.plot(x, np.sin(x))
plt.title('Синус')
plt.xlabel('x')
plt.ylabel('sin(x)')

plt.subplot(3,1,2)
plt.plot(x, np.sin(x),'-g', label='sin(x)')
plt.plot(x, np.cos(x),':b', label='cos(x)')

plt.title('Синус и косинус')
plt.xlabel('x')
plt.legend() # чтобы отображалась легенда на графике, после каждого сабплота нужен

plt.subplot(3,1,3)

plt.plot(x, np.sin(x),'-g', label='sin(x)')
plt.plot(x, np.cos(x),':b', label='cos(x)')

plt.title('Синус и косинус')
plt.xlabel('x')
plt.axis('equal')
plt.legend()

plt.subplots_adjust(hspace=0.5) # перед show

plt.show()

# маркеры можно совмещать со стилем линий
x = np.linspace(0,10,30)
plt.plot(x, np.sin(x), 'o', color='green')
plt.plot(x, np.sin(x)+1, '>', color='green')
plt.plot(x, np.sin(x)+2, '--^', color='green')
plt.plot(x, np.sin(x)+3, '--p', color='green', markersize=15,
         linewidth=4, markerfacecolor='white',
         markeredgecolor='gray',
         markeredgewidth=2)
plt.show()


rng=np.random.default_rng(0)
colors=rng.random(30)
sizes=100 * rng.random(30)
plt.scatter(x, np.sin(x), marker='o', c=colors, s=sizes) # можно задать для каждой точки отдельные характеристики
plt.colorbar()

# Если точек больше 1000, то предпочтительнее plot из-за производительности
plt.show()


x = np.linspace(0, 10, 50)
dy=0.4
y = np.sin(x) + dy * np.random.randn(50)

plt.errorbar(x, y, yerr=dy, fmt='.k') # график с погрешностью dy
plt.fill_between(x, y-dy, y+dy, color='red', alpha=0.4) # alpha - параметр прозрачности

plt.show()


def f(x,y):
    return np.sin(x)**5+np.cos(20+x*y)*np.cos(x)
x=np.linspace(0,5,50)
y=np.linspace(0,5,40)
X,Y=np.meshgrid(x,y)
Z=f(X,Y)

#plt.contour(X, Y,Z, color='red')
# plt.contour(X, Y,Z, cmap='RdGy') # cmap уже предустановленная цветовая схема
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian') # intepolation - сглаживание
plt.colorbar()

# по умолчанию начало координат в левом нижнем углу, можно менять
#plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='upper')

# aspect='equal' - соотношение сторон
plt.show()


# можно выводить вместе
plt.contour(X, Y,Z)
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='lower')
plt.colorbar()

plt.show()


c = plt.contour(X, Y,Z, color='red')
plt.clabel(c) # на контуре нарисовались значения
plt.imshow(Z, extent=[0,5,0,5], cmap='RdGy', interpolation='gaussian', origin='lower')
plt.colorbar()

plt.show()

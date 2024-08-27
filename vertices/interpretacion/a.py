import numpy as np
import matplotlib as plt
from mpl_toolkits.mplot3d import Axes3D
lado1 = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
lado2 = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
lado3 = np.array([[1, 1], [1, 2], [2, 2], [2, 1]])
grafo = {}
for nodo in lado1:
    nodo_x_y_z = np.append(nodo, -1)
    for cosa in lado2:
        if lado2[cosa][0] == nodo_x_y_z[0]:  #con x busco z
            grafo += [nodo_x_y_z[0], nodo_x_y_z[1], lado2[cosa][1]]
    for cosa in lado3:
        if lado3[cosa][0]== nodo_x_y_z[1]:  #con y busco z
            grafo += [nodo_x_y_z[0], nodo_x_y_z[1], lado3[cosa][1]]
            
#ploteamos el grafo 3d
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for key, value in grafo.items():
    x = [key[0], value[0]]
    y = [key[1], value[1]]
    z = [key[2], value[2]]
    ax.plot(x, y, z)

plt.show()


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np
from stl import mesh

ladoxy=[[1,1],[1,2],[2,1],[2,2]]
ladoxz=[[1,1],[1,2],[2,1],[2,2]]
ladoyz=[[1,1],[1,2],[2,1],[2,2]]

grafo=[]
conecciones=[[1,2],[2,3],[3,4],[4,1],[1,3],[2,4],[1,5],[2,6],[3,7],[4,8],[5,6],[6,7],[7,8],[8,5],[5,7],[6,8]]

""" conecciones gpt
conecciones = [
    [1, 2], [2, 4], [4, 3], [3, 1],  # Conexiones base inferior
    [5, 6], [6, 8], [8, 7], [7, 5],  # Conexiones base superior
    [1, 5], [2, 6], [3, 7], [4, 8],  # Conexiones verticales
    [1, 3], [2, 4], [5, 7], [6, 8]   # Conexiones diagonales para formar triángulos
] """


for nodo in ladoxy:
    for nodo2 in ladoxz:
        if nodo2[0]==nodo[0]:
            nodoaux=[nodo[0],nodo[1],nodo2[1]]
            if nodoaux not in grafo:
                grafo.append(nodoaux)

    for nodo3 in ladoyz:
        if nodo3[0]==nodo[1]:
            nodoaux=[nodo[0],nodo[1],nodo3[1]]
            if nodoaux not in grafo:
                grafo.append(nodoaux)

G = nx.Graph()
G.add_edges_from(conecciones)

triangles = []

for edge in conecciones:
    idx1 = edge[0] - 1 
    idx2 = edge[1] - 1
    point1 = grafo[idx1]
    point2 = grafo[idx2]
    
    """ recomendado por gpt
        if point1[2] != point2[2]:  # Si z es diferente, usar esa diferencia para formar el triángulo
            point3 = [point1[0], point2[1], point1[2]]
        else:  # De lo contrario, usar la diferencia en x o y
            point3 = [point1[0], point1[1], point2[2]] 
    """

    point3 = [point1[0], point1[1], point2[2]]
    triangles.append([point1, point2, point3])


mesher = np.zeros(len(triangles), dtype=mesh.Mesh.dtype)
for i, triangle in enumerate(triangles):
    mesher['vectors'][i] = np.array(triangle)


grafo_stl = mesh.Mesh(mesher)
grafo_stl.save('grafo_unificado.stl')


#grafico
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for nodo in grafo:
    x = nodo[0]
    y = nodo[1]
    z = nodo[2]
    ax.scatter(x, y, z)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
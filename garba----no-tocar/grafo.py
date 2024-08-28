import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


ladoxy=[[1,1],[2,2]]
ladoxz=[[1,1],[2,2]]
ladoyz=[[1,1],[2,2]]


grafo=[]

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

print(grafo)


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
import numpy as np
from stl import mesh
import os
import numpy as np
from typing import List
import copy  
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Tuple
from typing import List, Tuple



lado_xy=[[0,0],[1,0],[1,1],[0,1]] 
lado_yz=[[0,0],[1,0],[1,1],[0,1]]
lado_xz=[[0,0],[1,0],[1,1],[0,1]] 

conecciones_xy=[[1,3],[0,2],[1,3],[2,0]]
conecciones_yz=[[1,3],[0],[3],[2,0]]
conecciones_xz=[[1,3],[0,2],[1,3],[2,0]]
grafo=[]
padres = {}
    
for i, nodo in enumerate(lado_xy):
    for j, nodo2 in enumerate(lado_xz):
        if nodo2[0]==nodo[0]:
            for k, nodo3 in enumerate(lado_yz):
                if nodo3[1]==nodo[1] and nodo3[0]==nodo2[1]:
                    nodoaux=[nodo[0],nodo[1],nodo2[1]]
                    palabranodoaux=str(nodoaux)
                    padres[palabranodoaux]=[i,j,k]
                    if nodoaux not in grafo:
                        grafo.append(nodoaux)
                        
b=0
grafo_conexiones=[]
for i, nodo in enumerate(grafo):
    palabranodo = str(nodo)
    aux = padres[palabranodo]
    aux1 = aux[0]
    aux2 = aux[1]
    aux3 = aux[2]
    for nodo2 in grafo:
        if nodo == nodo2:
            continue
        
        palabranodo2 = str(nodo2)
        Aaux = padres[palabranodo2]
        Aaux1 = Aaux[0]
        Aaux2 = Aaux[1]
        Aaux3 = Aaux[2]
        si1 = False
        si2 = False
        si3 = False
        
        if Aaux1 in conecciones_xy[aux1]:
            si1 = True
        if Aaux2 in conecciones_xz[aux2]:
            si2 = True
        if Aaux3 in conecciones_yz[aux3]:
            si3 = True
        
        
        if (si1 and si2 and Aaux3==aux3) or (si1 and si3 and Aaux2 == aux2) or (si2 and si3 and Aaux1 == aux1)or (si1 and si2 and si3):
            b+=1
            #tenemos que agregar la coneccion de nodo a nodo2 en el grafo 3d
            if i not in grafo_conexiones:
                grafo_conexiones.append([])
            grafo_conexiones[i].append(nodo2)
        
            
print(b)


megagrafo=[grafo,grafo_conexiones]

def plot_megagrafo(megagrafo: Tuple[List[List[int]], List[List[List[int]]]]):
    grafo, grafo_conexiones = megagrafo
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot nodes
    for nodo in grafo:
        ax.scatter(nodo[0], nodo[1], nodo[2], c='b', marker='o')

    # Plot edges
    for i, conexiones in enumerate(grafo_conexiones):
        for conexion in conexiones:
            x = [grafo[i][0], conexion[0]]
            y = [grafo[i][1], conexion[1]]
            z = [grafo[i][2], conexion[2]]
            ax.plot(x, y, z, c='r')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

plot_megagrafo(megagrafo)
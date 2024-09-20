import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import List
from typing import Tuple
import numpy as np
import stl as mesh
import os


#ejemplo priamide
lado_xy=[[0,0],[1,0],[1,1],[0,1]] #nodos 1,2,3,4
lado_xz=[[0,0],[1,0],[1,1],[0,1]] #nodos 5,6,7,8
lado_yz=[[0,0],[1,0],[1,1],[0,1]] #nodos 9,10,11,12

conecciones_xy=[[1,3],[0,2],[1,3],[2,0]]
conecciones_yz=[[1,3],[0,2],[1,3],[2,0]]
conecciones_xz=[[1,3],[0,2],[1,3],[2,0]]
grafo=[]
grafo_conexiones=[]
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


for nodo in grafo:
    palabranodo=str(nodo)
    aux =padres[palabranodo]
    aux1=aux[0]
    aux2=aux[1]
    aux3=aux[2]
    for nodo2 in grafo:
        if nodo==nodo2:
            continue
        palabranodo2=str(nodo2)
        Aaux =padres[palabranodo2]
        Aaux1=Aaux[0]
        Aaux2=Aaux[1]
        Aaux3=Aaux[2]
        si1=False
        si2=False
        si3=False
        
        if Aaux1 in conecciones_xy[aux1]:
            si1=True
        if Aaux2 in conecciones_xz[aux2]:
            si2=True
        if Aaux3 in conecciones_yz[aux3]:
            si3=True
        if (si1 and si2) or (si1 and si3) or (si2 and si3):
            grafo_conexiones[nodo].append(nodo2)
            grafo_conexiones[nodo2].append(nodo) 
        
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# Plot nodes
for nodo in grafo:
    ax.scatter(nodo[0], nodo[1], nodo[2], c='b', marker='o')
# Plot connections
for nodo, conexiones in grafo_conexiones.items():
    for conexion in conexiones:
        xs = [nodo[0], conexion[0]]
        ys = [nodo[1], conexion[1]]
        zs = [nodo[2], conexion[2]]
        ax.plot(xs, ys, zs, c='r')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()


"""funcion para guardar el stl"""
def save_mesh(vertices: np.ndarray, faces: np.ndarray, filename: str) -> None:
    #armo el mesh
    mesh_data = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    
    #asigno los vertices a las caras
    for i, face in enumerate(faces):
        for j in range(3):
            mesh_data.vectors[i][j] = vertices[face[j]]
    
    #descargo el mesh en downloads
    downloads_path = os.path.expanduser("~/Downloads")
    full_path = os.path.join(downloads_path, filename)
    mesh_data.save(full_path)
    print(f'Mesh saved to {full_path}')



"""funcion para generar el mesh a partir de un grafo"""
def GenerarMeshFromGraph(graph: Tuple[List[List[int]], List[List[float]]]) -> None:
    vertices = graph[1]
    connections = graph[0]
    all_faces = []
    
    for i in range(len(vertices)):
        con = connections[i] 
        num_con = len(con)
        for j in range(num_con):
            v1 = i
            v2 = con[j]
            v3 = con[((j + 1) % num_con)]
            #checkeo de que no exista esta cara
            if [v1, v2, v3] not in all_faces and [v3, v2, v1] not in all_faces:
                all_faces.append([v1, v2, v3])
            
    
    all_faces = np.array(all_faces)
    all_vertices = np.array(vertices)
    
    save_mesh(all_vertices, all_faces, 'output1.stl')


graph= (conecciones, grafo)
GenerarMeshFromGraph(graph)
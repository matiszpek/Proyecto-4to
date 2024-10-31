from typing import List, Tuple
import numpy as np # type: ignore
from stl import mesh # type: ignore

def create_3d_object(coordinates):
    # Create a numpy array from the coordinates
    vertices = np.array(coordinates)

    # Create a mesh object
    mesh_object = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, vertex in enumerate(vertices):
        mesh_object.vectors[i] = vertex
    return mesh_object

def export_stl(mesh_object, filename):
    # Export the mesh object as an STL file
    mesh_object.save(filename)



def crearCaras(objeto):
    con, ver = objeto
    caras=[]

    for i in range(len(con)):
        for j in range(len(ver)):
            if j in con[i]:
                for k in range(len(ver)):
                    if k!=j and k in con[i]:
                        caras.append([i,j,k])
                        print(con[i],con[j],con[k])

                        if j not in con[k]:
                            con[j].append(k)
                        if k not in con[j]:
                            con[k].append(j)
                        
                        if i in con[j]:
                            con[j].remove(i)
                        if i in con[k]:
                            con[k].remove(i)
    return caras 

def llamarIndice(lista, coords):
    
    #in1: list[int]=[], in2: list[int]=[], in3: list[int]=[]
    out = []
    for i in lista:
        in1=coords[i[0]]
        in2=coords[i[1]]
        in3=coords[i[2]]
        out.append([in1,in2,in3])
        #print([in1,in2,in3])
    return out

def sacarIteraciones(caras):
    vistos= set()
    resultado= []
    for i in caras:
        if frozenset(i) not in vistos:
            resultado.append(i)
            vistos.add(frozenset(i))
    return resultado

def figure():
    """input("Vertices:\n")""" 
    """input("\nConections:\n")"""
    vertices: List[List[float]] = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    
    connections: List[List[int]] = [[1, 2], [5, 3], [0, 3, 6], [2, 7, 1], [0, 5, 6], [4, 7, 1], [2, 4, 7], [3, 6, 5]]

    return connections, vertices


if __name__=="__main__":

    ver= figure()[1]
    object=llamarIndice(sacarIteraciones(crearCaras(figure())),ver)
    object3D = create_3d_object(object)
    export_stl(object3D,"C:\\Users\\48389918\\Desktop\\ Output.stl")
    print("Objeto creado")
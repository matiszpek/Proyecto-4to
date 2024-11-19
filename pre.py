from typing import List, Tuple
from stl import mesh # type: ignore
from scipy.spatial import ConvexHull, Delaunay
from mpl_toolkits.mplot3d import Axes3D
import numpy as np # type: ignore
import matplotlib.pyplot as plt
import os

#Conseguimos la ruta del escritorio
desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
output_file = os.path.join(desktop_path, "Output.stl")
output_file2 = os.path.join(desktop_path, "OutputFalso.stl")
print(output_file)

#Funcion que crea un objeto stl 3D
def create_3d_object(coordinates):
    # Create a numpy array from the coordinates
    vertices = np.array(coordinates)

    # Create a mesh object
    mesh_object = mesh.Mesh(np.zeros(vertices.shape[0], dtype=mesh.Mesh.dtype))
    for i, vertex in enumerate(vertices):
        mesh_object.vectors[i] = vertex
    return mesh_object

#Funcion que exporta un objeto stl
def export_stl(mesh_object, filename):
    # Export the mesh object as an STL file
    mesh_object.save(filename)


#Crear las caras de un objeto
def crearCaras(objeto):
    con, ver = objeto
    caras=[]

    for i in range(len(con)):
        for j in range(len(ver)):
            if j in con[i]:
                for k in range(len(ver)):
                    if k!=j and k in con[i]:
                        caras.append([i,j,k])
                        #print(con[i],con[j],con[k])

                        if j not in con[k]:
                            con[j].append(k)
                        if k not in con[j]:
                            con[k].append(j)
                        
                        if i in con[j]:
                            con[j].remove(i)
                        if i in con[k]:
                            con[k].remove(i)
    return caras 

def crearCaras2(objeto):
    con, ver = objeto
    caras=[]
    for i in range(len(con)):
        for j in range(len(ver)):
            if j in con[i]:
                for k in range(len(ver)):
                    if k!=j and k in con[i]:
                        caras.append([i,j,k])
                        #print(con[i],con[j],con[k])
    return caras


#Convertir los numeros de las conecciones en coordenadas
def llamarIndice(lista, coords):
    
    out = []
    for i in lista:
        in1=coords[i[0]]
        in2=coords[i[1]]
        in3=coords[i[2]]
        out.append([in1,in2,in3])
        #print([in1,in2,in3])
    return out

#Sacar posibles repeticiones de las caras
def sacarIteraciones(caras):
    vistos= set()
    resultado= []
    for i in caras:
        if frozenset(i) not in vistos:
            resultado.append(i)
            vistos.add(frozenset(i))
    return resultado


#Funcion que contiene los vertices y sus conecciones
# vertices: List[List[float]] = [
#     
#     [0,0,0],
#     [-1,-1,0],
#     [-1,1,0],
#     [1,0,0],
#     [-1,0,1],
# ]
# 
# # Definimos las conexiones (aristas) entre los vértices del octaedro.
# connections = [
#     
#     [[-1,-1,0],[-1,1,0],[-1,0,1]],
#     [[-1,1,0],[1,0,0],[-1,0,1]],
#     [[1,0,0],[-1,-1,0],[-1,0,1]],
#     [[-1,-1,0],[-1,1,0],[-1,0,1]],
#     [[-1,-1,0],[1,0,0],[-1,1,0]]
# ]
# return connections, vertices


#Traducir las conecciones en vertices a numeros
def translate(figure):
    connections, vertices= figure
    for ver in range(len(vertices)):
        k=-1
        for i in connections:
            k+=1
            for j in range(len(i)):
                if vertices[ver]==connections[k][j]:
                    connections[k][j]=ver
    return connections, vertices


#Fuincion que se fija si objeto es convexo
def convex(vertices: List[List[float]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array(vertices)
    hull = ConvexHull(points)
    
    # Plot the vertices
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo')
    
    # Plot the edges of the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Cycle back to the first point
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    # Check if all points are on the convex hull
    hull_vertices = set(hull.vertices)
    all_points_on_hull = all(i in hull_vertices for i in range(len(points)))
    
    return all_points_on_hull

def main(coneciones, vertices):
    coneciones, vertices=translate(coneciones, vertices)

    if convex(vertices):#La figura es convexa
        print("La figura es convexa")
        
        object=llamarIndice(sacarIteraciones(crearCaras((coneciones, vertices))),vertices) #Crear las caras del objeto
        object3D = create_3d_object(object) #Crear el objeto 3D
        #export_stl(object3D, output_file) #Exportar el objeto
        export_stl(object3D, "output.stl") #Exportar el objeto|
        
        print("Objeto creado")
        
    else:
        print("La figura no es convexa")

        object=llamarIndice(sacarIteraciones(crearCaras2((coneciones, vertices))),vertices) #Crear las caras del objeto
        object3D = create_3d_object(object) #Crear el objeto 3D
        export_stl(object3D, output_file) #Exportar el objeto
        #export_stl(object3D, "output.stl") #Exportar el objeto|

        object2=llamarIndice(sacarIteraciones(crearCaras((coneciones, vertices))),vertices) #Crear las caras del objeto
        object3D2 = create_3d_object(object) #Crear el objeto 3D
        export_stl(object3D, output_file2) #Exportar el objeto|

        print("Objeto creado")

if __name__=="__main__":
    main(figure())
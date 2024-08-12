from stl import mesh
import numpy as np 
from index import caras
from index import to_stl

nombre=input("nombre:")
N = int(input("Número de puntos (vértices): "))
puntos = []
for i in range(N):
    x = float(input(f"Coordenada x del punto {i+1}: "))
    y = float(input(f"Coordenada y del punto {i+1}: "))
    z = float(input(f"Coordenada z del punto {i+1}: "))
    puntos.append([x, y, z])

M = int(input("Número de conexiones: "))                                                                                                                          
conecciones = []
for i in range(M):
    p1 = int(input(f"Índice del primer punto de la conexión {i+1}: "))
    p2 = int(input(f"Índice del segundo punto de la conexión {i+1}: "))
    conecciones.append((p1, p2))

car=caras(puntos,conecciones)
to_stl(puntos, conecciones, nombre)

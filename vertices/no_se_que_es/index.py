from stl import mesh
import numpy as np

def caras(puntos, conecciones):
    caras = []
    for p1, p2 in conecciones:
        cara = []
        for i, punto in enumerate(puntos):
            if i == p1:
                cara.append(punto)
            elif i == p2:
                cara.append(punto)
        caras.append(cara)
    return caras

def to_stl(puntos, conecciones, filename):
    caras = caras(puntos, conecciones)
    vertices = np.array(caras)
    faces = np.array([
        [0, 1, 2],
        [2, 3, 0]
    ])
    your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            your_mesh.vectors[i][j] = vertices[f[j]]
    your_mesh.save(filename)


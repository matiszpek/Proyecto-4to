import numpy as np
from stl import mesh

cara1 = []
cara2 = []
cara3 = []

graph = {
    'A': {'B', 'C', 'E', 'I'},  # Vértice A conectado a B (XY), C (XY), E (XZ), I (YZ)
    'B': {'A', 'D', 'F', 'J'},  # Vértice B conectado a A (XY), D (XY), F (XZ), J (YZ)
    'C': {'A', 'D', 'G', 'K'},  # Vértice C conectado a A (XY), D (XY), G (XZ), K (YZ)
    'D': {'B', 'C', 'H', 'L'},  # Vértice D conectado a B (XY), C (XY), H (XZ), L (YZ)
    'E': {'A', 'F', 'G'},       # Vértice E conectado a A (XZ-XY), F (XZ), G (XZ)
    'F': {'B', 'E', 'H'},       # Vértice F conectado a B (XZ-XY), E (XZ), H (XZ)
    'G': {'C', 'E', 'H'},       # Vértice G conectado a C (XZ-XY), E (XZ), H (XZ)
    'H': {'D', 'F', 'G'},       # Vértice H conectado a D (XZ-XY), F (XZ), G (XZ)
    'I': {'A', 'J', 'K'},       # Vértice I conectado a A (YZ-XY), J (YZ), K (YZ)
    'J': {'B', 'I', 'L'},       # Vértice J conectado a B (YZ-XY), I (YZ), L (YZ)
    'K': {'C', 'I', 'L'},       # Vértice K conectado a C (YZ-XY), I (YZ), L (YZ)
    'L': {'D', 'J', 'K'}        # Vértice L conectado a D (YZ-XY), J (YZ), K (YZ)
}
positions = {
    'A': (0, 30, 0),  # XY: (0, 30)
    'B': (75, 30, 0), # XY: (75, 30)
    'C': (0, 0, 0),   # XY: (0, 0)
    'D': (75, 0, 0),  # XY: (75, 0)
    'E': (0, 30, 9),  # XZ: (0, 9)
    'F': (75, 30, 9), # XZ: (75, 9)
    'G': (0, 0, 9),   # XZ: (0, 0)
    'H': (75, 0, 9),  # XZ: (75, 0)
    'I': (0, 9, 30),  # YZ: (30, 9)
    'J': (0, 18, 30), # YZ: (30, 18)
    'K': (0, 18, 0),  # YZ: (18, 0)
    'L': (0, 9, 0)    # YZ: (9, 0)
}


# Ejemplo de cómo se podrían definir algunas caras utilizando posiciones
vertices = np.array([
    positions['A'], positions['B'], positions['D'],
    positions['D'], positions['C'], positions['A']
    # Continuar con el resto de las caras
])

# Definir las caras usando los índices de los vértices
faces = np.array([
    [0, 1, 2],
    [2, 3, 0],
    
    # Continuar con el resto de las caras
])

# Crear la malla
your_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
for i, f in enumerate(faces):
    for j in range(3):
        your_mesh.vectors[i][j] = vertices[f[j]]

# Guardar la malla en un archivo STL
your_mesh.save('output.stl')

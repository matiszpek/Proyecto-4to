import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


# Supongamos que tienes vértices en 2D para las tres vistas: superior, frontal y lateral
vertices_superior = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])  # Ejemplo vista superior
vertices_frontal = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])   # Ejemplo vista frontal
vertices_lateral = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])   # Ejemplo vista lateral

# 1. Crear nodos por cada vértice
G = nx.Graph()

def add_vertices_to_graph(vertices, view_name):
    for i, v in enumerate(vertices):
        G.add_node(f"{view_name}_{i}", pos=v)

add_vertices_to_graph(vertices_superior, "superior")
add_vertices_to_graph(vertices_frontal, "frontal")
add_vertices_to_graph(vertices_lateral, "lateral")

# 2. Conectar nodos que están a la misma altura en diferentes vistas
def connect_views(view1, view2):
    for node1 in G.nodes(data=True):
        if node1[1]['view'] == view1:
            for node2 in G.nodes(data=True):
                if node2[1]['view'] == view2:
                    if node1[1]['pos'][0] == node2[1]['pos'][0]:
                        G.add_edge(node1[0], node2[0])

connect_views("superior", "frontal")
connect_views("superior", "lateral")
connect_views("frontal", "lateral")

# 3. Detección de ciclos de longitud 3
cycles = [cycle for cycle in nx.cycle_basis(G) if len(cycle) == 3]

# 4. Unificar los vértices correspondientes
unified_vertices = []
for cycle in cycles:
    # Aquí puedes calcular la posición 3D combinando las posiciones 2D en las diferentes vistas
    # Por simplicidad, tomaré el promedio
    positions = np.array([G.nodes[node]['pos'] for node in cycle])
    unified_vertex = np.mean(positions, axis=0)
    unified_vertices.append(unified_vertex)

# Ahora `unified_vertices` contiene la lista de los vértices unificados en 3D

# 5. Crear un grafo con los vértices unificados
G_unified = nx.Graph()
for i, v in enumerate(unified_vertices):
    G_unified.add_node(i, pos=v)

# 6. Conectar los nodos que estaban conectados en las vistas 2D
def connect_unified_views():
    for node1 in G_unified.nodes():
        for node2 in G_unified.nodes():
            if node1 != node2:
                pos1 = G_unified.nodes[node1]['pos']
                pos2 = G_unified.nodes[node2]['pos']
                if pos1[0] == pos2[0]:
                    G_unified.add_edge(node1, node2)

connect_unified_views()
connect_unified_views()
connect_unified_views()

# Ahora `G_unified` contiene el grafo con los vértices unificados en 3D y las conexiones correspondientes

# 7. Visualización del grafo
pos = nx.get_node_attributes(G_unified, 'pos')
nx.draw(G_unified, pos, with_labels=True, node_size=700, node_color='skyblue', font_size=10)
plt.show()

# 8. Exportar a formato de archivo
nx.write_adjlist(G_unified, "grafo_unificado.stl")

# Ahora tienes un archivo "grafo_unificado.stl" que contiene la información del grafo unificado en 3D
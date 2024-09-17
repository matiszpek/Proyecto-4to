def unificar_nodos(puntos_vista1, puntos_vista2, puntos_vista3, conexiones_vista1, conexiones_vista2, conexiones_vista3):
    """
    Función para unificar nodos que corresponden al mismo punto en 3D, pero que tienen diferentes identificadores.
    """
    # Mapa para identificar nodos unificados, con un identificador común
    vertices_3d = {}
    puntos_3d = []  # Lista para almacenar los puntos en 3D
    next_vertex_id = 0  # Para llevar un control de los índices de los vértices en 3D

    # Unificar nodos basados en coincidencia de coordenadas
    for i, (x1, y1) in enumerate(puntos_vista1):  # Recorrer puntos de la vista 1
        for j, (x2, z2) in enumerate(puntos_vista2):  # Recorrer puntos de la vista 2
            if x1 == x2:  # Coincidencia de X entre vista 1 y 2
                for k, (y3, z3) in enumerate(puntos_vista3):  # Recorrer puntos de la vista 3
                    if y1 == y3 and z2 == z3:  # Coincidencia de Y y Z entre vistas 1 y 3
                        # Agregar el vértice 3D a la lista de puntos
                        puntos_3d.append([x1, y1, z2])
                        # Mapear los identificadores de las vistas al mismo vértice en 3D
                        vertices_3d[(i, j, k)] = next_vertex_id
                        next_vertex_id += 1

    # Proyectar conexiones desde las tres vistas
    conexiones_3d = []

    # Conexiones en vista 1
    for (nodo1, nodo2) in conexiones_vista1:
        for key1, vertice1 in vertices_3d.items():
            if key1[0] == nodo1:  # Si el nodo coincide con vista 1
                for key2, vertice2 in vertices_3d.items():
                    if key2[0] == nodo2:  # Verificar el segundo nodo
                        conexiones_3d.append([vertice1, vertice2])

    # Conexiones en vista 2
    for (nodo1, nodo2) in conexiones_vista2:
        for key1, vertice1 in vertices_3d.items():
            if key1[1] == nodo1:  # Coincide en vista 2
                for key2, vertice2 in vertices_3d.items():
                    if key2[1] == nodo2:  # Verificar el segundo nodo
                        conexiones_3d.append([vertice1, vertice2])

    # Conexiones en vista 3
    for (nodo1, nodo2) in conexiones_vista3:
        for key1, vertice1 in vertices_3d.items():
            if key1[2] == nodo1:  # Coincide en vista 3
                for key2, vertice2 in vertices_3:
                    if key2[2] == nodo2:  # Verificar el segundo nodo
                        conexiones_3d.append([vertice1, vertice2])
    return puntos_3d, conexiones_3d

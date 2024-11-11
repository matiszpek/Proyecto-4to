import open3d as o3d
import numpy as np

# Paso 1: Crear una nube de puntos a partir de tu lista de puntos
points = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
])
nube_puntos = o3d.geometry.PointCloud()
nube_puntos.points = o3d.utility.Vector3dVector(points)

# Paso 2: Generar la malla con Poisson Surface Reconstruction
# Aumenta `depth` para obtener mayor detalle si es necesario
mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(nube_puntos, depth=8)

# Paso 3: Filtrar caras sueltas o con poca densidad si es necesario
vertices = np.asarray(mesh.vertices)
mesh.remove_vertices_by_mask(densities < np.quantile(densities, 0.1))

# Paso 4: Guardar o visualizar el modelo
o3d.io.write_triangle_mesh("modelo_concavo_poisson.stl", mesh)
o3d.visualization.draw_geometries([mesh])

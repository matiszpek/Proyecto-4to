import numpy as np
from stl import mesh

def sphere_sdf(p, center, radius):
    return np.linalg.norm(p - center) - radius

def box_sdf(p, center, size):
    q = np.abs(p - center) - size
    return np.linalg.norm(np.maximum(q, 0)) + np.minimum(np.max(q), 0)

def union_sdf(sdf1, sdf2, p):
    return np.minimum(sdf1(p), sdf2(p))

def evaluate_sdf(sdf, grid_size, bounds):
    x = np.linspace(bounds[0][0], bounds[0][1], grid_size)
    y = np.linspace(bounds[1][0], bounds[1][1], grid_size)
    z = np.linspace(bounds[2][0], bounds[2][1], grid_size)
    grid = np.meshgrid(x, y, z, indexing='ij')
    points = np.stack(grid, axis=-1)
    sdf_values = np.empty(points.shape[:-1])
    for ix in range(grid_size):
        for iy in range(grid_size):
            for iz in range(grid_size):
                sdf_values[ix, iy, iz] = sdf(points[ix, iy, iz])
    return sdf_values

def discretize_sdf_to_voxels(sdf_values, threshold=0):
    voxels = sdf_values < threshold
    return voxels

def voxels_to_mesh(voxels):
    vertices = []
    faces = []
    voxel_shape = voxels.shape
    for x in range(voxel_shape[0] - 1):
        for y in range(voxel_shape[1] - 1):
            for z in range(voxel_shape[2] - 1):
                if voxels[x, y, z]:
                    v0 = [x, y, z]
                    v1 = [x+1, y, z]
                    v2 = [x+1, y+1, z]
                    v3 = [x, y+1, z]
                    v4 = [x, y, z+1]
                    v5 = [x+1, y, z+1]
                    v6 = [x+1, y+1, z+1]
                    v7 = [x, y+1, z+1]
                    vertex_count = len(vertices)
                    vertices.extend([v0, v1, v2, v3, v4, v5, v6, v7])
                    faces.extend([
                        [vertex_count, vertex_count+1, vertex_count+2],
                        [vertex_count, vertex_count+2, vertex_count+3],
                        [vertex_count+4, vertex_count+5, vertex_count+6],
                        [vertex_count+4, vertex_count+6, vertex_count+7],
                        [vertex_count, vertex_count+1, vertex_count+5],
                        [vertex_count, vertex_count+5, vertex_count+4],
                        [vertex_count+1, vertex_count+2, vertex_count+6],
                        [vertex_count+1, vertex_count+6, vertex_count+5],
                        [vertex_count+2, vertex_count+3, vertex_count+7],
                        [vertex_count+2, vertex_count+7, vertex_count+6],
                        [vertex_count+3, vertex_count, vertex_count+4],
                        [vertex_count+3, vertex_count+4, vertex_count+7]
                    ])
    return np.array(vertices), np.array(faces)

def save_mesh_to_stl(vertices, faces, filename):
    data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    for i, f in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = vertices[f[j], :]
    m = mesh.Mesh(data)
    m.save(filename)

# Parámetros
grid_size = 50
bounds = [[-1, 1], [-1, 1], [-1, 1]]
sphere_center = np.array([0.3, 0.3, 0.3])
sphere_radius = 0.5
box_center = np.array([-0.3, -0.3, -0.3])
box_size = np.array([0.5, 0.5, 0.5])

# Definir las SDFs
sphere = lambda p: sphere_sdf(p, sphere_center, sphere_radius)
box = lambda p: box_sdf(p, box_center, box_size)
combined_sdf = lambda p: union_sdf(sphere, box, p)

# Evaluar SDF combinada
sdf_values = evaluate_sdf(combined_sdf, grid_size, bounds)

# Discretizar SDF a voxels
voxels = discretize_sdf_to_voxels(sdf_values)

# Convertir voxels a mesh
vertices, faces = voxels_to_mesh(voxels)

# Guardar mesh a archivo STL
save_mesh_to_stl(vertices, faces, 'output_combined.stl')
print("Archivo guardado como 'output_combined.stl'")

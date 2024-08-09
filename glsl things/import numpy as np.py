import numpy as np
from skimage import measure
from stl import mesh

def sdf_sphere(p, r):
    return np.linalg.norm(p) - r

def sdf_box(p, b):
    d = np.abs(p) - b
    return np.linalg.norm(np.maximum(d, 0), axis=-1) + np.minimum(np.maximum(d[..., 0], np.maximum(d[..., 1], d[..., 2])), 0)

def sdf_union(d1, d2):
    return np.minimum(d1, d2)

def map(p):
    sphere = sdf_sphere(p - np.array([0.0, 0.0, 0.0]), 0.3)
    box = sdf_box(p - np.array([0.25, 0.0, 0.0]), np.array([0.3, 0.3, 0.3]))
    return sdf_union(sphere, box)

def sample_sdf(grid_size=64, bounds=1.0):
    lin = np.linspace(-bounds, bounds, grid_size)
    x, y, z = np.meshgrid(lin, lin, lin)
    p = np.stack([x, y, z], axis=-1)
    d = map(p)
    return d

def generate_mesh(sdf_grid, level=0.0):
    verts, faces, normals, values = measure.marching_cubes(sdf_grid, level)
    return verts, faces

def save_mesh_as_stl(verts, faces, filename='output.stl'):
    # Create the mesh
    data = np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype)
    for i, f in enumerate(faces):
        for j in range(3):
            data['vectors'][i][j] = verts[f[j], :]
    m = mesh.Mesh(data)
    m.save(filename)
    print(f"Archivo guardado en: {filename}")

# Muestreamos la SDF en una cuadrícula 3D
sdf_grid = sample_sdf(grid_size=128)

# Generamos la malla utilizando Marching Cubes
verts, faces = generate_mesh(sdf_grid)

# Guardamos la malla como un archivo .stl en una ubicación específica
save_mesh_as_stl(verts, faces, 'output.stl')

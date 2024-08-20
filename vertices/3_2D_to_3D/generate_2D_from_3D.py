from stl import mesh
import matplotlib.pyplot as plt
import json
import random as r

file_name = 'cylinder.stl'
random_scale = 0.025

_m = mesh.Mesh.from_file('vertices/3_2D_to_3D/' + file_name)

figure = plt.figure()
figure_ = plt.figure()
obj_ax = figure.add_subplot(projection='3d')
x_ax = figure_.add_subplot(221)
y_ax = figure_.add_subplot(222)
z_ax = figure_.add_subplot(223)

if input('Do you want to add noise? [y/n] ') != 'y':
    random_scale = 0


x_points = list(set(list(zip([float(row[0] + r.random()*random_scale) for row in _m.x], [float(row[0] + r.random()*random_scale) for row in _m.y]))))
y_points = list(set(list(zip([float(row[0] + r.random()*random_scale) for row in _m.y], [float(row[0] + r.random()*random_scale) for row in _m.z]))))
z_points = list(set(list(zip([float(row[0] + r.random()*random_scale) for row in _m.z], [float(row[0] + r.random()*random_scale) for row in _m.x]))))

obj_ax.scatter3D(_m.x.flatten(), _m.y.flatten(), _m.z.flatten())

x_ax.scatter([row[0] for row in x_points], [row[1] for row in x_points])
y_ax.scatter([row[0] for row in y_points], [row[1] for row in y_points])
z_ax.scatter([row[0] for row in z_points], [row[1] for row in z_points])

# save the points to a file
with open('vertices/3_2D_to_3D/' + file_name[:-4] + '.json', 'w') as f:
    json.dump({'x': x_points, 'y': y_points, 'z': z_points}, f)

# Auto scale to the mesh size
scale = _m.points.flatten()
obj_ax.auto_scale_xyz(scale, scale, scale)

# Show the plot to the screen
if input('Do you want to display the plot? [y/n] ') == 'y':
    plt.show()
import matplotlib.pyplot as plt
import json

file_name = 'cylinder.json'

with open('vertices/3_2D_to_3D/' + file_name, 'r') as f:
    points = json.load(f)

figure = plt.figure()
x_ax = figure.add_subplot(221)
y_ax = figure.add_subplot(222)
z_ax = figure.add_subplot(223)

x_ax.scatter([row[0] for row in points['x']], [row[1] for row in points['x']])
y_ax.scatter([row[0] for row in points['y']], [row[1] for row in points['y']])
z_ax.scatter([row[0] for row in points['z']], [row[1] for row in points['z']])

plt.show()
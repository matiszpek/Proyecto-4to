import matplotlib.pyplot as plt
import json

file_name = 'cylinder.json'

with open('vertices/3_2D_to_3D/' + file_name, 'r') as f:
    points = json.load(f)

figure = plt.figure()

x_ax = figure.add_subplot(221)
y_ax = figure.add_subplot(222)
z_ax = figure.add_subplot(223)


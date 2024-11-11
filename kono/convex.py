from typing import List
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

#Fuincion que se fija si objeto es convexo
def convex(vertices: List[List[float]]):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    points = np.array(vertices)
    hull = ConvexHull(points)
    
    # Plot the vertices
    ax.plot(points[:, 0], points[:, 1], points[:, 2], 'bo')
    
    # Plot the edges of the convex hull
    for simplex in hull.simplices:
        simplex = np.append(simplex, simplex[0])  # Cycle back to the first point
        ax.plot(points[simplex, 0], points[simplex, 1], points[simplex, 2], 'r-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.show()
    # Check if all points are on the convex hull
    hull_vertices = set(hull.vertices)
    all_points_on_hull = all(i in hull_vertices for i in range(len(points)))
    
    return all_points_on_hull
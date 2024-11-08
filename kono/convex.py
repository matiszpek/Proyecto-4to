from scipy.spatial import ConvexHull, Delaunay
import matplotlib.pyplot as plt
import numpy as np

def convex(list):

    points=np.array([list])
    # Use the QJ option to joggle the input points
    hull = ConvexHull(points, qhull_options='QJ')
    delaunay = Delaunay(points, qhull_options='QJ')

    # Check if each point is inside the convex hull
    inside_hull = all(delaunay.find_simplex(points) >= 0)
    print("All points are inside the convex hull:", inside_hull)

    edges = list(zip(*points))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i in hull.simplices:
        plt.plot(points[i,0], points[i,1], points[i,2], 'r-')

    ax.plot(edges[0], edges[1], edges[2], 'bo') 

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()
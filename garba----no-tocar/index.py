from modelo import matrix as mx
from modelo import funciones as fn

def main():
    n = 100
    xy, xz, yz = mx.create_word_matrix("chona",n), mx.cuadrado(n), mx.cuadrado(n)
    matrix = fn.createMatrix(xy, xz, yz,n)
    fn.voxel_to_mesh(matrix)

if __name__ == "__main__":
    main()

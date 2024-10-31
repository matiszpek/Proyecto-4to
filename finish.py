from project import figure
from typing import List, Tuple
from megacodigo import todo


lado_xy=[[0,0],[1,0],[1,1],[0,1]] 
lado_yz=[[0,0],[1,0],[1,1],[0,1]]
lado_xz=[[0,0],[1,0],[1,1],[0,1]] 

conecciones_xy=[[1,3],[0,2],[1,3],[2,0]]
conecciones_yz=[[1,3],[0,2],[1,3],[2,0]]
conecciones_xz=[[1,3],[0,2],[1,3],[2,0]]
grafo=[]
padres = {}

todo(lado_xy, lado_yz, lado_xz, conecciones_xy, conecciones_yz, conecciones_xz)

crearCaras = figure()
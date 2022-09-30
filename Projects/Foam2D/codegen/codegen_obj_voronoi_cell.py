import casadi as ca
import os

# Problem dimensions
N = 20  # max number of neighbor sites + 2 (?)

# Input: Objective function parameters
area_weight = ca.MX.sym('area_weight', 1, 1)
length_weight = ca.MX.sym('length_weight', 1, 1)
centroid_weight = ca.MX.sym('centroid_weight', 1, 1)
area_target = ca.MX.sym('area_target', 1, 1)
num_neighbors = ca.MX.sym('num_neighbors', 1, 1)
p = ca.horzcat(area_weight, length_weight, centroid_weight, area_target, num_neighbors)

# Input: Voronoi sites
c = ca.MX.sym('c', 2, N)
xc, yc = ca.vertsplit(c)

idx = 1 + ca.mod(ca.transpose(ca.linspace(0, N - 1, N)), num_neighbors)

x1 = xc[0]
y1 = yc[0]
x2 = xc[idx[:-1]]
y2 = yc[idx[:-1]]
x3 = xc[idx[1:]]
y3 = yc[idx[1:]]

m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1))
xn = 0.5 * (x1 + x3) - m * (y3 - y1)
yn = 0.5 * (y1 + y3) + m * (x3 - x1)

x2 = xn
y2 = yn
x3 = ca.horzcat(xn[1:], xn[0])
y3 = ca.horzcat(yn[1:], yn[0])

idx = ca.transpose(ca.linspace(0, 18, 19))

Obj = ca.MX.zeros(1, 1)

# Area objective
a = ca.if_else(
    idx < num_neighbors,
    0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)),
    0
)
A = ca.sum2(a)
Obj += area_weight * (A - area_target) * (A - area_target)

# Perimeter minimization objective
l2 = ca.if_else(
    idx < num_neighbors,
    (x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2),
    0
)
L = ca.sum2(l2)
Obj += length_weight * L

# Centroid objective
tx = ca.if_else(
    idx < num_neighbors,
    a * (x1 + x2 + x3) / 3,
    0
)
ty = ca.if_else(
    idx < num_neighbors,
    a * (y1 + y2 + y3) / 3,
    0
)
px = ca.sum2(tx) / A
py = ca.sum2(ty) / A
D = (x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)
Obj += centroid_weight * D

# Generate and compile C code
opts = dict(with_header=True)

ca_O_voronoi_cell = ca.Function('ca_O_voronoi_cell', [c, p], [Obj])
ca_O_voronoi_cell.generate('ca_O_voronoi_cell', opts)
print('compiling generated code for voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_O_voronoi_cell.c -o libca_O_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_O_voronoi_cell.c -o ca_O_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dOdc_voronoi_cell = ca.Function('ca_dOdc_voronoi_cell', [c, p], [ca.jacobian(Obj, c)])
ca_dOdc_voronoi_cell.generate('ca_dOdc_voronoi_cell', opts)
print('compiling generated code for gradient of voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_voronoi_cell.c -o libca_dOdc_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_voronoi_cell.c -o ca_dOdc_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_d2Odc2_voronoi_cell = ca.Function('ca_d2Odc2_voronoi_cell', [c, p], [ca.hessian(Obj, c)[0]])
ca_d2Odc2_voronoi_cell.generate('ca_d2Odc2_voronoi_cell', opts)
print('compiling generated code for hessian of voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_voronoi_cell.c -o libca_d2Odc2_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_voronoi_cell.c -o ca_d2Odc2_voronoi_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

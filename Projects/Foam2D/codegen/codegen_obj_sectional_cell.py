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

# Input: sectional sites
c = ca.MX.sym('c', 3, N)
xc, yc, zc = ca.vertsplit(c)

idx = 1 + ca.mod(ca.transpose(ca.linspace(0, N - 1, N)), num_neighbors)

x1 = xc[0]
y1 = yc[0]
z1 = zc[0]
x2 = xc[idx[:-1]]
y2 = yc[idx[:-1]]
z2 = zc[idx[:-1]]
x3 = xc[idx[1:]]
y3 = yc[idx[1:]]
z3 = zc[idx[1:]]

m2 = -(y2 - y1) / (x2 - x1)
c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 * z2 - z1 * z1) / (2 * (x2 - x1))
m3 = -(y3 - y1) / (x3 - x1)
c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 * z3 - z1 * z1) / (2 * (x3 - x1))

yn = (c3 - c2) / (m2 - m3)
xn = m2 * yn + c2

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

ca_O_sectional_cell = ca.Function('ca_O_sectional_cell', [c, p], [Obj])
ca_O_sectional_cell.generate('ca_O_sectional_cell', opts)
print('compiling generated code for sectional objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_O_sectional_cell.c -o libca_O_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_O_sectional_cell.c -o ca_O_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dOdc_sectional_cell = ca.Function('ca_dOdc_sectional_cell', [c, p], [ca.jacobian(Obj, c)])
ca_dOdc_sectional_cell.generate('ca_dOdc_sectional_cell', opts)
print('compiling generated code for gradient of sectional objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_sectional_cell.c -o libca_dOdc_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_sectional_cell.c -o ca_dOdc_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_d2Odc2_sectional_cell = ca.Function('ca_d2Odc2_sectional_cell', [c, p], [ca.hessian(Obj, c)[0]])
ca_d2Odc2_sectional_cell.generate('ca_d2Odc2_sectional_cell', opts)
print('compiling generated code for hessian of sectional objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_sectional_cell.c -o libca_d2Odc2_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_sectional_cell.c -o ca_d2Odc2_sectional_cell.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

import casadi as ca
import os
from codegen_obj_base import obj_base, gen_code

# Problem dimensions
N = 20  # max number of neighbor sites + 2 (?)

# Input: Objective function parameters
area_weight = ca.MX.sym('area_weight', 1, 1)
length_weight = ca.MX.sym('length_weight', 1, 1)
centroid_weight = ca.MX.sym('centroid_weight', 1, 1)
area_target = ca.MX.sym('area_target', 1, 1)
num_neighbors = ca.MX.sym('num_neighbors', 1, 1)
p = ca.horzcat(area_weight, length_weight, centroid_weight, area_target, num_neighbors)

# Input: power sites
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
c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 - z1) / (2 * (x2 - x1))
m3 = -(y3 - y1) / (x3 - x1)
c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 - z1) / (2 * (x3 - x1))

yn = (c3 - c2) / (m2 - m3)
xn = m2 * yn + c2

Obj = obj_base(x1, y1, x2, y2, xn, yn, area_weight, length_weight, centroid_weight, area_target, num_neighbors)

# Generate and compile C code
ident = 'power_cell'
gen_code(ident, c, p, Obj)

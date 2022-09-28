import casadi as ca
import os

# Problem dimensions
Nc_fixed = 40  # number of fixed sites
Nc_free = 40  # number of free sites
Nt = 118  # max number of triangles in dual triangulation
Na = 300  # max number of area triangles

# Input: Objective function parameters
area_weight = ca.MX.sym('area_weight', 1, 1)
length_weight = ca.MX.sym('length_weight', 1, 1)
centroid_weight = ca.MX.sym('centroid_weight', 1, 1)
area_target = ca.MX.sym('area_target', 1, Nc_free)
p = ca.horzcat(area_weight, length_weight, centroid_weight, area_target)

# Input: Voronoi sites
c_fixed = ca.MX.sym('c_fixed', 2, Nc_fixed)
c_free = ca.MX.sym('c_free', 2, Nc_free)
c = ca.horzcat(c_free, c_fixed)
xc, yc = ca.vertsplit(c)

# Input: Dual triangulation vertex indices
tri = ca.MX.sym('tri', 3, Nt)
v1, v2, v3 = ca.vertsplit(tri)

# Input: Cell edge node indices
e = ca.MX.sym('e', 3, Na)
vc, ve1, ve2 = ca.vertsplit(e)

x1 = xc[v1]
y1 = yc[v1]
x2 = xc[v2]
y2 = yc[v2]
x3 = xc[v3]
y3 = yc[v3]

m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1))
xn = 0.5 * (x1 + x3) - m * (y3 - y1)
yn = 0.5 * (y1 + y3) + m * (x3 - x1)

x1 = xc[vc]
y1 = yc[vc]
x2 = xn[ve1]
y2 = yn[ve1]
x3 = xn[ve2]
y3 = yn[ve2]

Obj = ca.MX.zeros(1, 1)

# Area objective
a = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
for i in range(Nc_free):
    A = ca.sum2(a * (vc == i))
    Obj += area_weight * (A - area_target[i]) * (A - area_target[i])

# Perimeter minimization objective
l2 = (x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2)
L = ca.sum2(l2 * (ve1 > ve2) * (vc < Nc_free))
Obj += length_weight * L

# Centroidal objective
d = (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2)
D = ca.sum2(d * (vc < Nc_free))
Obj += centroid_weight * D

# Generate and compile C code
opts = dict(with_header=True)

ca_O_voronoi = ca.Function('ca_O_voronoi', [c_free, c_fixed, tri, e, p], [Obj])
ca_O_voronoi.generate('ca_O_voronoi', opts)
print('compiling generated code for voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_O_voronoi.c -o libca_O_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_O_voronoi.c -o ca_O_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dOdc_voronoi = ca.Function('ca_dOdc_voronoi', [c_free, c_fixed, tri, e, p], [ca.jacobian(Obj, c_free)])
ca_dOdc_voronoi.generate('ca_dOdc_voronoi', opts)
print('compiling generated code for gradient of voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_voronoi.c -o libca_dOdc_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dOdc_voronoi.c -o ca_dOdc_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_d2Odc2_voronoi = ca.Function('ca_d2Odc2_voronoi', [c_free, c_fixed, tri, e, p], [ca.hessian(Obj, c_free)[0]])
ca_d2Odc2_voronoi.generate('ca_d2Odc2_voronoi', opts)
print('compiling generated code for hessian of voronoi objective function...')
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_voronoi.c -o libca_d2Odc2_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_voronoi.c -o ca_d2Odc2_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

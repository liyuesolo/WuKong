import casadi as ca
import os

# problem dimensions
Nc = 80  # number of Voronoi sites
Nt = 118  # number of Delaunay triangles

# Input: Voronoi sites with z offset from plane
c = ca.MX.sym('c', 3, Nc)
xc, yc, zc = ca.vertsplit(c)

# Input: Triangle vertex indices
tri = ca.MX.sym('tri', 3, Nt)
v1, v2, v3 = ca.vertsplit(tri)

x1 = xc[v1]
y1 = yc[v1]
z1 = zc[v1]
x2 = xc[v2]
y2 = yc[v2]
z2 = zc[v2]
x3 = xc[v3]
y3 = yc[v3]
z3 = zc[v3]

m2 = -(y2 - y1) / (x2 - x1)
c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 * z2 - z1 * z1) / (2 * (x2 - x1))
m3 = -(y3 - y1) / (x3 - x1)
c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 * z3 - z1 * z1) / (2 * (x3 - x1))

yn = (c3 - c2) / (m2 - m3)
xn = m2 * yn + c2

# Output: Sectional voronoi nodes
x = ca.vec(ca.vertcat(xn, yn))

# Generate and compile C code
opts = dict(with_header=True)

ca_x_sectional = ca.Function('ca_x_sectional', [c, tri], [x])
ca_x_sectional.generate('ca_x_sectional', opts)
print('compiling generated code for voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_x_sectional.c -o libca_x_sectional.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_x_sectional.c -o ca_x_sectional.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dxdc_sectional = ca.Function('ca_dxdc_sectional', [c, tri], [ca.jacobian(x, c)])
ca_dxdc_sectional.generate('ca_dxdc_sectional', opts)
print('compiling generated code for gradient of voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_dxdc_sectional.c -o libca_dxdc_sectional.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dxdc_sectional.c -o ca_dxdc_sectional.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

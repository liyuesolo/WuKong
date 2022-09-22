import casadi as ca
import os

# problem dimensions
Nc = 80  # number of Voronoi sites
Nt = 118  # number of Delaunay triangles

# Input: Voronoi sites
c = ca.MX.sym('c', 2, Nc)
xc, yc = ca.vertsplit(c)

# Input: Triangle vertex indices
tri = ca.MX.sym('tri', 3, Nt)
v1, v2, v3 = ca.vertsplit(tri)

x1 = xc[v1]
y1 = yc[v1]
x2 = xc[v2]
y2 = yc[v2]
x3 = xc[v3]
y3 = yc[v3]

m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1))
xn = 0.5 * (x1 + x3) - m * (y3 - y1)
yn = 0.5 * (y1 + y3) + m * (x3 - x1)

# Output: Voronoi nodes (i.e. circumcenters of Delaunay triangles)
x = ca.vec(ca.vertcat(xn, yn))

# Generate and compile C code
opts = dict(with_header=True)

ca_x = ca.Function('ca_x', [c, tri], [x])
ca_x.generate('ca_x', opts)
print('compiling generated code for voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_x.c -o libca_x.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_x.c -o ca_x.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dxdc = ca.Function('ca_dxdc', [c, tri], [ca.jacobian(x, c)])
ca_dxdc.generate('ca_dxdc', opts)
print('compiling generated code for gradient of voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_dxdc.c -o libca_dxdc.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dxdc.c -o ca_dxdc.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

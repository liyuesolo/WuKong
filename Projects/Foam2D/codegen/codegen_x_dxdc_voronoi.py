import casadi as ca
import os

# Problem dimensions
Nc_fixed = 40  # number of fixed sites
Nc_free = 40  # number of free sites
Nt = 118  # number of Delaunay triangles

# Input: Voronoi sites
c_fixed = ca.MX.sym('c_fixed', 2, Nc_fixed)
c_free = ca.MX.sym('c_free', 2, Nc_free)
c = ca.horzcat(c_free, c_fixed)
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

ca_x_voronoi = ca.Function('ca_x_voronoi', [c_free, c_fixed, tri], [x])
ca_x_voronoi.generate('ca_x_voronoi', opts)
print('compiling generated code for voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_x_voronoi.c -o libca_x_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_x_voronoi.c -o ca_x_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dxdc_voronoi = ca.Function('ca_dxdc_voronoi', [c_free, c_fixed, tri], [ca.jacobian(x, c_free)])
ca_dxdc_voronoi.generate('ca_dxdc_voronoi', opts)
print('compiling generated code for gradient of voronoi nodes...')
cmd = 'gcc -fPIC -shared -O0 ca_dxdc_voronoi.c -o libca_dxdc_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dxdc_voronoi.c -o ca_dxdc_voronoi.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

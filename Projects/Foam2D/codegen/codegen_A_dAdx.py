import casadi as ca
import os

# problem dimensions
Nc = 40  # number of Voronoi site areas (excludes convex hull sites)
Nx = 118  # number of Voronoi nodes
Na = 300  # max number of area triangles

# Input: Voronoi sites
c = ca.MX.sym('c', 2, Nc)
xc, yc = ca.vertsplit(c)

# Input: Voronoi nodes
x = ca.MX.sym('x', 2, Nx)
xn, yn = ca.vertsplit(x)

# Input: Area triangle vertex indices
e = ca.MX.sym('e', 3, Na)
vc, vx1, vx2 = ca.vertsplit(e)

# Output: Voronoi cell areas
x1 = xc[vc]
y1 = yc[vc]
x2 = xn[vx1]
y2 = yn[vx1]
x3 = xn[vx2]
y3 = yn[vx2]
a = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

A = ca.MX.zeros(Nc, 1)
for i in range(Nc):
    A[i] = ca.sum2(a * (vc == i))

# Generate and compile C code
opts = dict(with_header=True)

ca_A = ca.Function('ca_A', [c, x, e], [A])
ca_A.generate('ca_A', opts)
print('compiling generated code for voronoi cell areas...')
cmd = 'gcc -fPIC -shared -O0 ca_A.c -o libca_A.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_A.c -o ca_A.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dAdx = ca.Function('ca_dAdx', [c, x, e], [ca.jacobian(A, x)])
ca_dAdx.generate('ca_dAdx', opts)
print('compiling generated code for gradient of voronoi cell areas...')
cmd = 'gcc -fPIC -shared -O0 ca_dAdx.c -o libca_dAdx.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dAdx.c -o ca_dAdx.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

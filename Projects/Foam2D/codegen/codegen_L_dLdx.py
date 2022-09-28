import casadi as ca
import os

# Problem dimensions
Nc = 40  # number of Voronoi sites (excludes convex hull sites)
Nx = 118  # number of Voronoi nodes
Na = 300  # max number of edges

# Input: Voronoi nodes
x = ca.MX.sym('x', 2, Nx)
xn, yn = ca.vertsplit(x)

# Input: Area triangle vertex indices
e = ca.MX.sym('e', 3, Na)
vc, vx1, vx2 = ca.vertsplit(e)

# Output: Voronoi cell areas
x1 = xn[vx1]
y1 = yn[vx1]
x2 = xn[vx2]
y2 = yn[vx2]
l = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)

L = ca.sum2(l * (vx1 > vx2))

# Generate and compile C code
opts = dict(with_header=True)

ca_L = ca.Function('ca_L', [x, e], [L])
ca_L.generate('ca_L', opts)
print('compiling generated code for voronoi perimeter...')
cmd = 'gcc -fPIC -shared -O0 ca_L.c -o libca_L.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_L.c -o ca_L.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

ca_dLdx = ca.Function('ca_dLdx', [x, e], [ca.jacobian(L, x)])
ca_dLdx.generate('ca_dLdx', opts)
print('compiling generated code for gradient of voronoi perimeter...')
cmd = 'gcc -fPIC -shared -O0 ca_dLdx.c -o libca_dLdx.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))
cmd = 'gcc -fPIC -shared -O0 ca_dLdx.c -o ca_dLdx.so'
status = os.system(cmd)
if status != 0:
    raise Exception('Command {} failed'.format(cmd))

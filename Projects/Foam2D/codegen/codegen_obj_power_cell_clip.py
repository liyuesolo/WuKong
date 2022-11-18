import casadi as ca
import os
from codegen_obj_base_clip import obj_base, gen_code


def obj_power_cell_node_ss(x0, y0, z0, xc1, yc1, zc1, xc2, yc2, zc2):
    x1 = x0
    y1 = y0
    z1 = z0
    x2 = xc1
    y2 = yc1
    z2 = zc1
    x3 = xc2
    y3 = yc2
    z3 = zc2

    rsq2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    d2 = 0.5 + 0.5 * (z2 - z1) / (rsq2 + 1e-14)
    xp2 = x1 + d2 * (x2 - x1)
    yp2 = y1 + d2 * (y2 - y1)
    xl2 = -(y2 - y1)
    yl2 = (x2 - x1)
    rsq3 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1)
    d3 = 0.5 + 0.5 * (z3 - z1) / (rsq3 + 1e-14)
    xp3 = x1 + d3 * (x3 - x1)
    yp3 = y1 + d3 * (y3 - y1)
    xl3 = -(y3 - y1)
    yl3 = (x3 - x1)

    a2 = (yl3 * (xp3 - xp2) - xl3 * (yp3 - yp2)) / (xl2 * yl3 - xl3 * yl2 + 1e-14)
    xn = xp2 + a2 * xl2
    yn = yp2 + a2 * yl2

    return [xn, yn]


def obj_power_cell_node_sb(x0, y0, z0, xc1, yc1, zc1, xbs2, ybs2, xbe2, ybe2):
    rsq = (xc1 - x0) * (xc1 - x0) + (yc1 - y0) * (yc1 - y0)
    d = 0.5 + 0.5 * (zc1 - z0) / (rsq + 1e-14)

    x1 = d * xc1 + (1 - d) * x0
    y1 = d * yc1 + (1 - d) * y0
    x2 = x1 + (yc1 - y0)
    y2 = y1 - (xc1 - x0)
    x3 = xbs2
    y3 = ybs2
    x4 = xbe2
    y4 = ybe2

    t = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / (-(x4 - x3) * (y2 - y1) + (x2 - x1) * (y4 - y3) + 1e-14)
    xn = x1 + t * (x2 - x1)
    yn = y1 + t * (y2 - y1)

    return [xn, yn]


def obj_power_cell_node_bb(xbs2, ybs2):
    return [xbs2, ybs2]


def codegen_obj_power_cell(N, opt=3):
    # Problem dimensions
    # N = 20  # max number of neighbor sites + 2 (?)

    # Input: Objective function parameters
    p = ca.MX.sym('p', 1, 8)

    # Input: Neighbor indices and type (cell or boundary edge)
    n = ca.MX.sym('n', 2, N + 3)
    nt, ni = ca.vertsplit(n)

    # Input: Voronoi sites
    c = ca.MX.sym('c', 3, N + 1)
    xc, yc, zc = ca.vertsplit(c)

    # Input: Boundary edges
    b = ca.MX.sym('b', 4, N + 1)
    xbs, ybs, xbe, ybe = ca.vertsplit(b)

    i1 = ni[1:-1]
    i2 = ni[2:]
    t1 = nt[1:-1]
    t2 = nt[2:]

    x0 = xc[0]
    y0 = yc[0]
    z0 = zc[0]

    xc1 = xc[i1]
    yc1 = yc[i1]
    zc1 = zc[i1]
    xc2 = xc[i2]
    yc2 = yc[i2]
    zc2 = zc[i2]

    xbs1 = xbs[i1]
    ybs1 = ybs[i1]
    xbe1 = xbe[i1]
    ybe1 = ybe[i1]

    xbs2 = xbs[i2]
    ybs2 = ybs[i2]
    xbe2 = xbe[i2]
    ybe2 = ybe[i2]

    nss = obj_power_cell_node_ss(x0, y0, z0, xc1, yc1, zc1, xc2, yc2, zc2)
    nsb = obj_power_cell_node_sb(x0, y0, z0, xc1, yc1, zc1, xbs2, ybs2, xbe2, ybe2)
    nbs = obj_power_cell_node_sb(x0, y0, z0, xc2, yc2, zc2, xbs1, ybs1, xbe1, ybe1)
    nbb = obj_power_cell_node_bb(xbs2, ybs2)

    type = t1 * 2 + t2
    xn = ca.vertcat(nss[0], nsb[0], nbs[0], nbb[0])[4 * ca.transpose(ca.linspace(0, N, N + 1)) + type]
    yn = ca.vertcat(nss[1], nsb[1], nbs[1], nbb[1])[4 * ca.transpose(ca.linspace(0, N, N + 1)) + type]

    Obj = obj_base(N, x0, y0, xn, yn, p)

    # Generate and compile C code
    ident = 'power_cell_' + str(N)
    gen_code(ident, p, n, c, b, Obj, opt)


if __name__ == "__main__":
    codegen_obj_power_cell(20)

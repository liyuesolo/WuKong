import casadi as ca
import os
from codegen_imagematch import *
from codegen_base import gen_code


def voronoi_cell_node_ss(x0, y0, xc1, yc1, xc2, yc2):
    x1 = x0
    y1 = y0
    x2 = xc1
    y2 = yc1
    x3 = xc2
    y3 = yc2

    m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1) + 1e-14)
    xn = 0.5 * (x1 + x3) - m * (y3 - y1)
    yn = 0.5 * (y1 + y3) + m * (x3 - x1)

    return [xn, yn]


def voronoi_cell_node_sb(x0, y0, xc1, yc1, xbs2, ybs2, xbe2, ybe2):
    x1 = (x0 + xc1) / 2
    y1 = (y0 + yc1) / 2
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


def voronoi_cell_node_bb(xbs2, ybs2):
    return [xbs2, ybs2]


def codegen_imagematch_voronoi_cell(MAX_N, MAX_P, opt=3):
    # Input: Objective function parameters
    p = ca.MX.sym('p', 1, 2)
    num_neighbors = p[0]
    num_points = p[1]

    # Input: Neighbor indices and type (cell or boundary edge)
    n = ca.MX.sym('n', 1, MAX_N + 1)

    # Input: Voronoi sites
    c = ca.MX.sym('c', 2, MAX_N + 1)
    xc, yc = ca.vertsplit(c)

    # Input: Pixel coordinates
    pix = ca.MX.sym('pix', 2, MAX_P)
    xp, yp = ca.vertsplit(pix)

    # Input: Boundary edges
    b = ca.MX.sym('b', 4, MAX_N + 1)
    xbs, ybs, xbe, ybe = ca.vertsplit(b)

    i1 = 1 + ca.mod(ca.transpose(ca.linspace(0, MAX_N, MAX_N + 1)), num_neighbors)
    i2 = ca.horzcat(i1[1:], i1[0])
    t1 = n[i1]
    t2 = n[i2]

    x0 = xc[0]
    y0 = yc[0]

    xc1 = xc[i1]
    yc1 = yc[i1]
    xc2 = xc[i2]
    yc2 = yc[i2]

    xbs1 = xbs[i1]
    ybs1 = ybs[i1]
    xbe1 = xbe[i1]
    ybe1 = ybe[i1]

    xbs2 = xbs[i2]
    ybs2 = ybs[i2]
    xbe2 = xbe[i2]
    ybe2 = ybe[i2]

    nss = voronoi_cell_node_ss(x0, y0, xc1, yc1, xc2, yc2)
    nsb = voronoi_cell_node_sb(x0, y0, xc1, yc1, xbs2, ybs2, xbe2, ybe2)
    nbs = voronoi_cell_node_sb(x0, y0, xc2, yc2, xbs1, ybs1, xbe1, ybe1)
    nbb = voronoi_cell_node_bb(xbs2, ybs2)

    type = t1 * 2 + t2
    xn = ca.vertcat(nss[0], nsb[0], nbs[0], nbb[0])[4 * ca.transpose(ca.linspace(0, MAX_N, MAX_N + 1)) + type]
    yn = ca.vertcat(nss[1], nsb[1], nbs[1], nbb[1])[4 * ca.transpose(ca.linspace(0, MAX_N, MAX_N + 1)) + type]

    Obj = imagematch(MAX_N, MAX_P, num_neighbors, num_points, xn, yn, xp, yp)

    # Generate and compile C code
    ident = 'imagematch_voronoi_cell_' + str(MAX_N)
    gen_code(ident, [p, n, c, b, pix], c, Obj, opt, 1)

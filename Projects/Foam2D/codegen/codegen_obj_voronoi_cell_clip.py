import casadi as ca
import numpy as np
import os
from codegen_obj_base_clip import obj_base, gen_code


def obj_voronoi_cell_node_ss(x0, y0, xc1, yc1, xc2, yc2):
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


def obj_voronoi_cell_node_sb(x0, y0, xc1, yc1, xbs2, ybs2, xbe2, ybe2):
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


def obj_voronoi_cell_node_bb(xbs2, ybs2):
    return [xbs2, ybs2]


def codegen_obj_voronoi_cell(N, opt=3):
    # Problem dimensions
    # N = 20  # max number of neighbor sites + 2 (?)

    # Input: Objective function parameters
    p = ca.MX.sym('p', 1, 8)
    num_neighbors = p[4]

    # Input: Neighbor indices and type (cell or boundary edge)
    n = ca.MX.sym('n', 2, N + 3)
    nt, ni = ca.vertsplit(n)

    # Input: Voronoi sites
    c = ca.MX.sym('c', 2, N + 1)
    xc, yc = ca.vertsplit(c)

    # Input: Boundary edges
    b = ca.MX.sym('b', 4, N + 1)
    xbs, ybs, xbe, ybe = ca.vertsplit(b)

    i1 = ni[1:-1]
    i2 = ni[2:]
    t1 = nt[1:-1]
    t2 = nt[2:]

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

    nss = obj_voronoi_cell_node_ss(x0, y0, xc1, yc1, xc2, yc2)
    nsb = obj_voronoi_cell_node_sb(x0, y0, xc1, yc1, xbs2, ybs2, xbe2, ybe2)
    nbs = obj_voronoi_cell_node_sb(x0, y0, xc2, yc2, xbs1, ybs1, xbe1, ybe1)
    nbb = obj_voronoi_cell_node_bb(xbs2, ybs2)

    type = t1 * 2 + t2
    xn = ca.vertcat(nss[0], nsb[0], nbs[0], nbb[0])[4 * ca.transpose(ca.linspace(0, N, N + 1)) + type]
    yn = ca.vertcat(nss[1], nsb[1], nbs[1], nbb[1])[4 * ca.transpose(ca.linspace(0, N, N + 1)) + type]

    # xn = ca.conditional(t1 * 2 + t2, [
    #     nss[0],
    #     nsb[0],
    #     nbs[0]],
    #                     nbb[0], False)
    # yn = ca.conditional(t1 * 2 + t2, [
    #     nss[1],
    #     nsb[1],
    #     nbs[1]],
    #                     nbb[1], False)

    # xn = nss[0]
    # yn = nss[1]

    Obj = obj_base(N, x0, y0, xn, yn, p)

    # Generate and compile C code
    ident = 'voronoi_cell_' + str(N)
    gen_code(ident, p, n, c, b, Obj, opt)


def obj_base_test(N, x1, y1, xn, yn, p):
    area_weight = p[0]
    length_weight = p[1]
    centroid_weight = p[2]
    area_target = p[3]
    num_neighbors = p[4]
    drag_target_weight = p[5]
    drag_x = p[6]
    drag_y = p[7]

    x2 = xn[:-1]
    y2 = yn[:-1]
    x3 = xn[1:]
    y3 = yn[1:]

    idx = np.linspace(0, N - 1, N)

    Obj = 0

    # Target position objective
    # Obj += drag_target_weight * ((x1 - drag_x) * (x1 - drag_x) + (y1 - drag_y) * (y1 - drag_y))

    # Area objective
    a = ca.if_else(
        idx < num_neighbors,
        0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)),
        0
    )
    A = np.sum(a)
    Obj += area_weight * (A - area_target) * (A - area_target) / area_target

    # Perimeter minimization objective
    l2 = ca.if_else(
        idx < num_neighbors,
        ca.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2) + 1e-14),
        0
    )
    L = np.sum(l2)
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
    px = np.sum(tx) / A
    py = np.sum(ty) / A
    D = (x1 - px) * (x1 - px) + (y1 - py) * (y1 - py)
    Obj += centroid_weight * D

    # Soft constraints
    Obj += 1e-14 / ca.constpow(A, 5)  # Prevent area going to zero
    # d = ca.if_else(
    #     idx < num_neighbors,
    #     1e-12 / ca.constpow((xb - x1) * (xb - x1) + (yb - y1) * (yb - y1), 2),
    #     0
    # )
    # Obj += ca.sum2(d)  # Prevent sites from getting too close. TODO: I broke this, might need to restore.

    return Obj


if __name__ == "__main__":
    # codegen_obj_voronoi_cell(20)

    # Input: Objective function parameters
    p = np.array([0.1, 0.003, 0.05, 0.05, 4, 0, 2.42715e-312, 6.94702e-310])

    # Input: Neighbor indices and type (cell or boundary edge)
    n = np.array([[0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 2, 3, 4, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    nt = n[0, :]
    ni = n[1, :]

    # Input: Voronoi sites
    c = np.array([[0.452138, 0, 0.646815, 0.537078, 0.203733, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [-0.664131, 0, -0.562264, -0.570056, -0.552297, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    xc = c[0, :]
    yc = c[1, :]

    # Input: Boundary edges
    b = np.array([[0, -0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, -0.75, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    xbs = b[0, :]
    ybs = b[1, :]
    xbe = b[2, :]
    ybe = b[3, :]

    i1 = ni[1:-1]
    i2 = ni[2:]
    t1 = nt[1:-1]
    t2 = nt[2:]

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

    nss = obj_voronoi_cell_node_ss(x0, y0, xc1, yc1, xc2, yc2)
    nsb = obj_voronoi_cell_node_sb(x0, y0, xc1, yc1, xbs2, ybs2, xbe2, ybe2)
    nbs = obj_voronoi_cell_node_sb(x0, y0, xc2, yc2, xbs1, ybs1, xbe1, ybe1)
    nbb = obj_voronoi_cell_node_bb(xbs2, ybs2)

    idx = t1 * 2 + t2
    print(ca.transpose(ca.horzcat(nss[0], nsb[0], nbs[0], nbb[0]))[
              4 * ca.transpose(ca.linspace(0, 20, 21)) + ca.transpose(idx)])
    xn = ca.horzcat(nss[0], nsb[0], nbs[0], nbb[0])[np.arange(21) + 21 * idx]
    yn = ca.horzcat(nss[1], nsb[1], nbs[1], nbb[1])[np.arange(21) + 21 * idx]

    # xn = np.zeros(21)
    # yn = np.zeros(21)
    # for i in range(21):
    #     if t1[i] * 2 + t2[i] < 0.5:
    #         xn[i] = nss[0][i]
    #         yn[i] = nss[1][i]
    #     elif t1[i] * 2 + t2[i] < 1.5:
    #         xn[i] = nsb[0][i]
    #         yn[i] = nsb[1][i]
    #     elif t1[i] * 2 + t2[i] < 2.5:
    #         xn[i] = nbs[0][i]
    #         yn[i] = nbs[1][i]
    #     else:
    #         xn[i] = nbb[0][i]
    #         yn[i] = nbb[1][i]

    print(xn)

    Obj = obj_base_test(20, x0, y0, xn, yn, p)
    print(Obj)

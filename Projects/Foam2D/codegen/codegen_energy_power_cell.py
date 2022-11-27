import casadi as ca
import os
from codegen_energy import energy
from codegen_base import gen_code


def codegen_energy_power_cell(N, opt=3):
    # Problem dimensions
    # N = 20  # max number of neighbor sites + 2 (?)

    # Input: Objective function parameters
    p = ca.MX.sym('p', 1, 8)
    num_neighbors = p[4]

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

    rsq2 = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)
    d2 = 0.5 + 0.5 * (z2 - z1) / rsq2
    xp2 = x1 + d2 * (x2 - x1)
    yp2 = y1 + d2 * (y2 - y1)
    xl2 = -(y2 - y1)
    yl2 = (x2 - x1)
    rsq3 = (x3 - x1) * (x3 - x1) + (y3 - y1) * (y3 - y1)
    d3 = 0.5 + 0.5 * (z3 - z1) / rsq3
    xp3 = x1 + d3 * (x3 - x1)
    yp3 = y1 + d3 * (y3 - y1)
    xl3 = -(y3 - y1)
    yl3 = (x3 - x1)

    a2 = (yl3 * (xp3 - xp2) - xl3 * (yp3 - yp2)) / (xl2 * yl3 - xl3 * yl2)
    xn = xp2 + a2 * xl2
    yn = yp2 + a2 * yl2

    # m2 = -(y2 - y1) / (x2 - x1)
    # c2 = (x2 * x2 - x1 * x1 + y2 * y2 - y1 * y1 + z2 - z1) / (2 * (x2 - x1))
    # m3 = -(y3 - y1) / (x3 - x1)
    # c3 = (x3 * x3 - x1 * x1 + y3 * y3 - y1 * y1 + z3 - z1) / (2 * (x3 - x1))
    #
    # yn = (c3 - c2) / (m2 - m3)
    # xn = m2 * yn + c2

    Obj = energy(N, x1, y1, xn, yn, p)

    # Generate and compile C code
    ident = 'energy_power_cell_' + str(N)
    gen_code(ident, [c, p], c, Obj, opt)

import casadi as ca
import os
from codegen_energy import energy
from codegen_base import gen_code


def codegen_energy_voronoi_cell(N, opt=3):
    # Input: Objective function parameters
    p = ca.MX.sym('p', 1, 8)
    num_neighbors = p[4]

    # Input: Voronoi sites
    c = ca.MX.sym('c', 2, N)
    xc, yc = ca.vertsplit(c)

    idx = 1 + ca.mod(ca.transpose(ca.linspace(0, N - 1, N)), num_neighbors)

    x1 = xc[0]
    y1 = yc[0]
    x2 = xc[idx[:-1]]
    y2 = yc[idx[:-1]]
    x3 = xc[idx[1:]]
    y3 = yc[idx[1:]]

    m = 0.5 * ((y3 - y2) * (y2 - y1) + (x3 - x2) * (x2 - x1)) / ((y3 - y1) * (x2 - x1) - (y2 - y1) * (x3 - x1))
    xn = 0.5 * (x1 + x3) - m * (y3 - y1)
    yn = 0.5 * (y1 + y3) + m * (x3 - x1)

    Obj = energy(N, x1, y1, xn, yn, p)

    # Generate and compile C code
    ident = 'energy_voronoi_cell_' + str(N)
    gen_code(ident, [c, p], c, Obj, opt)

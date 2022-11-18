import casadi as ca
import os


def obj_base(N, x1, y1, xn, yn, p):
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

    idx = ca.transpose(ca.linspace(0, N - 1, N))

    Obj = ca.MX.zeros(1, 1)

    # Target position objective
    Obj += drag_target_weight * ((x1 - drag_x) * (x1 - drag_x) + (y1 - drag_y) * (y1 - drag_y))

    # Area objective
    a = ca.if_else(
        idx < num_neighbors,
        0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)),
        0
    )
    A = ca.sum2(a)
    Obj += area_weight * (A - area_target) * (A - area_target) / area_target

    # Perimeter minimization objective
    l2 = ca.if_else(
        idx < num_neighbors,
        ca.sqrt((x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2) + 1e-14),
        0
    )
    L = ca.sum2(l2)
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
    px = ca.sum2(tx) / A
    py = ca.sum2(ty) / A
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


def gen_code(ident, p, n, c, b, Obj, opt=3):
    opts = dict(with_header=True)

    ca_O = ca.Function('ca_O_{}'.format(ident), [p, n, c, b], [Obj])
    ca_O.generate('ca_O_{}'.format(ident), opts)
    print('compiling generated code for {} objective function...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_O_{}.c -o libca_O_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_O_{}.c -o ca_O_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_dOdc = ca.Function('ca_dOdc_{}'.format(ident), [p, n, c, b], [ca.jacobian(Obj, c)])
    ca_dOdc.generate('ca_dOdc_{}'.format(ident), opts)
    print('compiling generated code for gradient of {} objective function...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_dOdc_{}.c -o libca_dOdc_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_dOdc_{}.c -o ca_dOdc_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_d2Odc2 = ca.Function('ca_d2Odc2_{}'.format(ident), [p, n, c, b], [ca.hessian(Obj, c)[0]])
    ca_d2Odc2.generate('ca_d2Odc2_{}'.format(ident), opts)
    print('compiling generated code for hessian of {} objective function...'.format(ident))
    cmd = 'gcc -fPIC -shared -O{} ca_d2Odc2_{}.c -o libca_d2Odc2_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O{} ca_d2Odc2_{}.c -o ca_d2Odc2_{}.so'.format(opt, ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

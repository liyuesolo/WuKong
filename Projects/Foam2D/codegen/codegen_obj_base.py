import casadi as ca
import os


def obj_base(x1, y1, xn, yn, area_weight, length_weight, centroid_weight, area_target, num_neighbors):
    x2 = xn
    y2 = yn
    x3 = ca.horzcat(xn[1:], xn[0])
    y3 = ca.horzcat(yn[1:], yn[0])

    idx = ca.transpose(ca.linspace(0, 18, 19))

    Obj = ca.MX.zeros(1, 1)

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
        (x3 - x2) * (x3 - x2) + (y3 - y2) * (y3 - y2),
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

    return Obj


def gen_code(ident, c, p, Obj):
    opts = dict(with_header=True)

    ca_O = ca.Function('ca_O_{}'.format(ident), [c, p], [Obj])
    ca_O.generate('ca_O_{}'.format(ident), opts)
    print('compiling generated code for power objective function...')
    cmd = 'gcc -fPIC -shared -O0 ca_O_{}.c -o libca_O_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O0 ca_O_{}.c -o ca_O_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_dOdc = ca.Function('ca_dOdc_{}'.format(ident), [c, p], [ca.jacobian(Obj, c)])
    ca_dOdc.generate('ca_dOdc_{}'.format(ident), opts)
    print('compiling generated code for gradient of power objective function...')
    cmd = 'gcc -fPIC -shared -O0 ca_dOdc_{}.c -o libca_dOdc_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O0 ca_dOdc_{}.c -o ca_dOdc_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

    ca_d2Odc2 = ca.Function('ca_d2Odc2_{}'.format(ident), [c, p], [ca.hessian(Obj, c)[0]])
    ca_d2Odc2.generate('ca_d2Odc2_{}'.format(ident), opts)
    print('compiling generated code for hessian of power objective function...')
    cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_{}.c -o libca_d2Odc2_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))
    cmd = 'gcc -fPIC -shared -O0 ca_d2Odc2_{}.c -o ca_d2Odc2_{}.so'.format(ident, ident)
    status = os.system(cmd)
    if status != 0:
        raise Exception('Command {} failed'.format(cmd))

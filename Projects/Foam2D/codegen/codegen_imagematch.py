import casadi as ca
import os


def imagematch(MAX_N, MAX_P, num_neighbors, num_points, xn, yn, xp, yp):
    x1 = xn[:-1]
    y1 = yn[:-1]
    x2 = xn[1:]
    y2 = yn[1:]

    n_idx = ca.transpose(ca.linspace(0, MAX_N - 1, MAX_N))
    p_idx = ca.linspace(0, MAX_P - 1, MAX_P)

    x1 = ca.repmat(x1, MAX_P, 1)
    y1 = ca.repmat(y1, MAX_P, 1)
    x2 = ca.repmat(x2, MAX_P, 1)
    y2 = ca.repmat(y2, MAX_P, 1)
    n_idx = ca.repmat(n_idx, MAX_P, 1)
    xp = ca.repmat(ca.transpose(xp), 1, MAX_N)
    yp = ca.repmat(ca.transpose(yp), 1, MAX_N)
    p_idx = ca.repmat(p_idx, 1, MAX_N)

    Obj = ca.MX.zeros(1, 1)

    x12 = x2 - x1
    y12 = y2 - y1
    x1p = xp - x1
    y1p = yp - y1

    # Target position objective
    cross = ca.if_else(
        ca.logic_and(n_idx < num_neighbors, p_idx < num_points),
        x12 * y1p - x1p * y12,
        100
    )
    obj = ca.constpow(ca.log(ca.exp(cross * -1.0) + 1), 2)

    Obj += ca.sum1(ca.sum2(obj))

    return Obj

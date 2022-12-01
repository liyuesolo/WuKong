import casadi as ca
import os


def imagematch(MAX_N, MAX_P, num_neighbors, num_points, xn, yn, xp, yp):
    x1 = xn[:-1]
    y1 = yn[:-1]
    x2 = xn[1:]
    y2 = yn[1:]

    n_idx = ca.repmat(ca.transpose(ca.linspace(0, MAX_N - 1, MAX_N)), 1, MAX_P)
    p_idx = ca.reshape(ca.repmat(ca.transpose(ca.linspace(0, MAX_P - 1, MAX_P)), MAX_N, 1), 1, MAX_N * MAX_P)

    x1 = ca.repmat(x1, 1, MAX_P)
    y1 = ca.repmat(y1, 1, MAX_P)
    x2 = ca.repmat(x2, 1, MAX_P)
    y2 = ca.repmat(y2, 1, MAX_P)
    xp = ca.reshape(ca.repmat(xp, MAX_N, 1), 1, MAX_N * MAX_P)
    yp = ca.reshape(ca.repmat(yp, MAX_N, 1), 1, MAX_N * MAX_P)

    Obj = ca.MX.zeros(1, 1)

    x12 = x2 - x1
    y12 = y2 - y1
    x1p = xp - x1
    y1p = yp - y1

    # Target position objective
    cross = ca.if_else(
        ca.logic_and(n_idx < num_neighbors, p_idx < 1),
        (x12 * y1p - x1p * y12) / ca.sqrt(x12 * x12 + y12 * y12 + 1e-14),
        0
    )
    # obj = ca.constpow(ca.log(ca.exp(cross * -1.0) + 1), 2)
    obj = ca.log(ca.exp(cross * -1.0) + 1)
    #
    obj = ca.if_else(
        ca.logic_and(n_idx < num_neighbors, p_idx < 1),
        (x12 * y1p - x1p * y12) / ca.sqrt(x12 * x12 + y12 * y12 + 1e-14),
        0
    )

    obj = ca.if_else(
        ca.logic_and(n_idx < num_neighbors, p_idx < 1),
        (x1p * x1p + y1p * y1p),
        0
    )
    # obj = ca.if_else(
    #     n_idx < 2,
    #     1 + 0.00000001 * x12,
    #     0
    # )

    Obj += ca.sum2(obj)
    # Obj -= x12[0, 0] * x12[0, 0]
    # Obj -= x12[0, 1] * x12[0, 1]
    # Obj -= x12[1, 0] * x12[1, 0]
    # Obj -= x12[1, 1] * x12[1, 1]

    return Obj


if __name__ == "__main__":
    MAX_N = 3
    MAX_P = 7
    n_idx = ca.repmat(ca.transpose(ca.linspace(0, MAX_N - 1, MAX_N)), 1, MAX_P)
    p_idx = ca.reshape(ca.repmat(ca.transpose(ca.linspace(0, MAX_P - 1, MAX_P)), MAX_N, 1), 1, MAX_N * MAX_P)

    print("N")
    print(n_idx)
    print("P")
    print(p_idx)

    A = ca.if_else(
        ca.logic_and(n_idx == 1, p_idx == 1),
        1,
        0
    )

    print(A)

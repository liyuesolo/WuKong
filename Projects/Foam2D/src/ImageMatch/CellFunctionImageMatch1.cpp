#include "../../include/ImageMatch/CellFunctionImageMatch1.h"
#include <iostream>

void CellFunctionImageMatch1::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                       const VectorXi &btype, double &value,
                                       const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        for (int i = 0; i < n_nodes; i++) {
            x0i = i * nx + 0;
            y0i = i * nx + 1;
            x1i = next(i) * nx + 0;
            y1i = next(i) * nx + 1;

            x0 = nodes(x0i);
            y0 = nodes(y0i);
            x1 = nodes(x1i);
            y1 = nodes(y1i);

            value += 0.1e1 / beta * log(0.1e1 + exp(-beta * ((x1 - x0) * (yp - y0) - (xp - x0) * (y1 - y0)) *
                                                    pow(pow(x1 - x0, 0.2e1) + pow(y1 - y0, 0.2e1) + epsilon,
                                                        -0.1e1 / 0.2e1)));
        }
    }
}

void CellFunctionImageMatch1::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                          const VectorXi &btype,
                                          VectorXT &gradient_c,
                                          VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    double t1, t2, t3, t4, t5, t6, t7, t8;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        for (int i = 0; i < n_nodes; i++) {
            x0i = i * nx + 0;
            y0i = i * nx + 1;
            x1i = next(i) * nx + 0;
            y1i = next(i) * nx + 1;

            x0 = nodes(x0i);
            y0 = nodes(y0i);
            x1 = nodes(x1i);
            y1 = nodes(y1i);

            t1 = x1 - x0;
            t2 = -y1 + y0;
            t3 = pow(t1, 0.2e1) + pow(t2, 0.2e1) + epsilon;
            t4 = pow(t3, -0.3e1 / 0.2e1);
            t3 = t3 * t4;
            t5 = yp - y0;
            t6 = xp - x0;
            t7 = t1 * t5 + t2 * t6;
            t4 = t7 * t4;
            t1 = t4 * t1;
            t7 = exp(-beta * t7 * t3);
            t8 = 0.1e1 + t7;
            t2 = t4 * t2;
            t4 = 0.1e1 / t8;
            gradient_x(x0i) += -(-(yp - y1) * t3 + t1) * t7 * t4;
            gradient_x(y0i) += (-t3 * (-x1 + xp) + t2) * t7 * t4;
            gradient_x(x1i) += (-t3 * t5 + t1) * t7 * t4;
            gradient_x(y1i) += -(-t3 * t6 + t2) * t7 * t4;

        }
    }
}

void CellFunctionImageMatch1::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                         const VectorXi &btype,
                                         MatrixXT &hessian,
                                         const CellInfo *cellInfo) const {
    int n_pix = cellInfo->border_pix.rows() / 2;
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double xp, yp;
    int x0i, y0i, x1i, y1i;
    double x0, y0, x1, y1;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21, t22, t23, t24, t25, t26, t27, t28, t29;
    for (int p = 0; p < n_pix; p++) {
        xp = cellInfo->border_pix(p * 2 + 0);
        yp = cellInfo->border_pix(p * 2 + 1);

        for (int i = 0; i < n_nodes; i++) {
            x0i = i * nx + 0;
            y0i = i * nx + 1;
            x1i = next(i) * nx + 0;
            y1i = next(i) * nx + 1;

            x0 = nodes(x0i);
            y0 = nodes(y0i);
            x1 = nodes(x1i);
            y1 = nodes(y1i);

            t1 = yp - y1;
            t2 = x1 - x0;
            t3 = -y1 + y0;
            t4 = pow(t3, 0.2e1);
            t5 = pow(t2, 0.2e1);
            t6 = t4 + t5 + epsilon;
            t7 = pow(t6, -0.5e1 / 0.2e1);
            t8 = t6 * t7;
            t9 = yp - y0;
            t10 = xp - x0;
            t11 = t2 * t9;
            t12 = t10 * t3;
            t13 = t12 + t11;
            t5 = 0.3e1 * t5 * t7;
            t14 = (-t5 + t8) * t13;
            t6 = pow(t6, 0.2e1) * t7;
            t15 = beta * t13;
            t16 = exp(-t15 * t6);
            t17 = 0.1e1 + t16;
            t18 = t13 * t8;
            t19 = t18 * t2;
            t20 = beta * (t1 * t6 - t19);
            t17 = 0.1e1 / t17;
            t16 = t16 * t17;
            t17 = t16 - 0.1e1;
            t21 = t16 / beta;
            t22 = -x1 + xp;
            t23 = t22 * t2;
            t24 = t1 * t3;
            t25 = beta * t8;
            t26 = 0.3e1 * t15 * t7;
            t27 = t26 * t2 * t3;
            t18 = t18 * t3;
            t28 = beta * (t22 * t6 - t18);
            t16 = -t16 + 0.1e1;
            t29 = t21 * (t20 * t28 * t16 + t25 * (t23 + t24) - t27);
            t19 = beta * (t6 * t9 - t19);
            t5 = t21 * (t20 * t19 * t16 + t25 * (t2 * (t1 + t9) + t13) - t15 * t5);
            t15 = t10 * t2;
            t18 = beta * (t10 * t6 - t18);
            t24 = t21 * (t20 * t18 * t16 - beta * (-t8 * (t24 + t15) + t6) - t27);
            t7 = (-0.3e1 * t4 * t7 + t8) * t13;
            t9 = t9 * t3;
            t6 = t21 * (t28 * t19 * t16 + beta * (t8 * (t23 + t9) + t6) - t27);
            t4 = t21 * (t28 * t18 * t16 - t26 * t4 + t25 * (t3 * (t10 + t22) + t13));
            t9 = t21 * (t19 * t18 * t16 + t25 * (t15 + t9) - t27);
            hess_xx(x0i, x0i) += -t21 * (t17 * pow(t20, 0.2e1) - beta * (0.2e1 * t1 * t8 * t2 + t14));
            hess_xx(x0i, y0i) += -t29;
            hess_xx(x0i, x1i) += -t5;
            hess_xx(x0i, y1i) += t24;
            hess_xx(y0i, x0i) += -t29;
            hess_xx(y0i, y0i) += -t21 * (t17 * pow(t28, 0.2e1) - beta * (0.2e1 * t22 * t8 * t3 + t7));
            hess_xx(y0i, x1i) += t6;
            hess_xx(y0i, y1i) += -t4;
            hess_xx(x1i, x0i) += -t5;
            hess_xx(x1i, y0i) += t6;
            hess_xx(x1i, x1i) += -t21 * (t17 * pow(t19, 0.2e1) - beta * (0.2e1 * t11 * t8 + t14));
            hess_xx(x1i, y1i) += -t9;
            hess_xx(y1i, x0i) += t24;
            hess_xx(y1i, y0i) += -t4;
            hess_xx(y1i, x1i) += -t9;
            hess_xx(y1i, y1i) += t21 * (t16 * pow(t18, 0.2e1) + beta * (0.2e1 * t12 * t8 + t7));
        }
    }
}

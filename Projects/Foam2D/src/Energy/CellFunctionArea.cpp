#include "../../include/Energy/CellFunctionArea.h"
#include <iostream>

void
CellFunctionArea::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                           const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, x1, y1, r;
    for (int i = 0; i < n_nodes; i++) {
        x0 = nodes(i * nx + 0);
        y0 = nodes(i * nx + 1);
        x1 = nodes(next(i) * nx + 0);
        y1 = nodes(next(i) * nx + 1);
        r = nodes(i * nx + 2);

        value += 0.5 * (x0 * y1 - x1 * y0);
        if (fabs(r) > 1e-10) {
            value += r * r * (0.2e1 * asin(sqrt(pow(y1 - y0, 0.2e1) + pow(x1 - x0, 0.2e1)) / r / 0.2e1) -
                              sin(0.2e1 * asin(sqrt(pow(y1 - y0, 0.2e1) + pow(x1 - x0, 0.2e1)) / r / 0.2e1))) / 0.2e1;
        }
    }
}

void
CellFunctionArea::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_c,
                              VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        ri = i * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        r = nodes(ri);

        gradient_x(x0i) += 0.5 * y1;
        gradient_x(y0i) += -0.5 * x1;
        gradient_x(x1i) += -0.5 * y0;
        gradient_x(y1i) += 0.5 * x0;

        if (fabs(r) > 1e-10) {
            t1 = -y1 + y0;
            t2 = x1 - x0;
            t3 = pow(t1, 0.2e1) + pow(t2, 0.2e1);
            t4 = pow(t3, -0.1e1 / 0.2e1);
            t5 = 0.1e1 / r;
            t6 = pow(t5, 0.2e1);
            t7 = -t3 * t6 + 0.4e1;
            t7 = pow(t7, -0.1e1 / 0.2e1);
            t3 = t3 * t4;
            t8 = 0.2e1 * asin(t3 * t5 / 0.2e1);
            t9 = cos(t8) - 0.1e1;
            t4 = t4 * t5;
            t5 = r * r;
            t2 = t5 * t4 * t2 * t7 * t9;
            t1 = t5 * t4 * t1 * t7 * t9;
            gradient_x(x0i) += t2;
            gradient_x(y0i) += -t1;
            gradient_x(x1i) += -t2;
            gradient_x(y1i) += t1;
            gradient_x(ri) += r * (r * t3 * t6 * t7 * t9 + t8 - sin(t8));
        }
    }
}

void
CellFunctionArea::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
                             const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, x1, y1, r;
    int x0i, y0i, x1i, y1i, ri;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20,
            t21, t22, t23, t24, t25, t26, t27, t28, t29, t30, t31, t32, t33, t34;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        ri = i * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        r = nodes(ri);

        hess_xx(x0i, y1i) += 0.5;
        hess_xx(y0i, x1i) += -0.5;
        hess_xx(x1i, y0i) += -0.5;
        hess_xx(y1i, x0i) += 0.5;

        if (fabs(r) > 1e-10) {
            t1 = -y1 + y0;
            t2 = x1 - x0;
            t3 = pow(t1, 0.2e1);
            t4 = pow(t2, 0.2e1);
            t5 = t3 + t4;
            t6 = pow(t5, -0.3e1 / 0.2e1);
            t7 = 0.1e1 / r;
            t8 = pow(t7, 0.2e1);
            t9 = t7 * t8;
            t10 = -t5 * t8 + 0.4e1;
            t11 = pow(t10, -0.3e1 / 0.2e1);
            t12 = t10 * t11;
            t13 = t5 * t6;
            t14 = pow(t5, 0.2e1);
            t15 = t14 * t6;
            t16 = 0.2e1;
            t17 = t16 * asin(t15 * t7 / 0.2e1);
            t18 = cos(t17);
            t19 = sin(t17);
            t20 = -t18 + 0.1e1;
            t21 = t18 - 0.1e1;
            t22 = t13 * t21;
            t23 = t20 * t6;
            t10 = 0.1e1 / t10;
            t24 = 0.1e1 / t5;
            t24 = 0.4e1 * t24 * t8;
            t4 = -t16 * t7 * (t12 * (t23 * t4 + t22) + t22 * t4 * t8 * t11) + t24 * t4 * t10 * t19;
            t25 = t22 * t8 * t11;
            t26 = t7 * t2;
            t27 = t16 * t26 * t1 * (t23 * t12 + t25) - t24 * t2 * t10 * t1 * t19;
            t18 = -t18 + 0.1e1;
            t28 = t13 * t12;
            t20 = t20 * t8 * t15 * t11 + t28 * t20;
            t29 = 0.4e1 * t9;
            t2 = t29 * t2 * t10 * t19 + t16 * t2 * t20 * t8;
            t13 = t16 * t26 * t13 * t12 * t18;
            t26 = r / 0.2e1;
            t30 = r * (t26 * t2 - t13);
            t31 = r * r / 0.2e1;
            t32 = t31 * t27;
            t33 = -t31 * t4;
            t34 = -t31 * t27;
            t3 = -t16 * t7 * (t12 * (t23 * t3 + t22) + t25 * t3) + t24 * t3 * t10 * t19;
            t20 = -t29 * t1 * t10 * t19 - t1 * t16 * t20 * t8;
            t1 = t16 * t28 * t7 * t1 * t18;
            t22 = r * (t26 * t20 + t1);
            t23 = -t31 * t27;
            t24 = -t31 * t3;
            t2 = r * (-t26 * t2 + t13);
            t13 = t31 * t27;
            t1 = r * (-t26 * t20 - t1);
            hess_xx(x0i, x0i) += t31 * t4;
            hess_xx(x0i, y0i) += t32;
            hess_xx(x0i, x1i) += t33;
            hess_xx(x0i, y1i) += t34;
            hess_xx(x0i, ri) += t30;
            hess_xx(y0i, x0i) += t32;
            hess_xx(y0i, y0i) += t31 * t3;
            hess_xx(y0i, x1i) += t23;
            hess_xx(y0i, y1i) += t24;
            hess_xx(y0i, ri) += t22;
            hess_xx(x1i, x0i) += t33;
            hess_xx(x1i, y0i) += t23;
            hess_xx(x1i, x1i) += t31 * t4;
            hess_xx(x1i, y1i) += t13;
            hess_xx(x1i, ri) += t2;
            hess_xx(y1i, x0i) += t34;
            hess_xx(y1i, y0i) += t24;
            hess_xx(y1i, x1i) += t13;
            hess_xx(y1i, y1i) += t31 * t3;
            hess_xx(y1i, ri) += t1;
            hess_xx(ri, x0i) += t30;
            hess_xx(ri, y0i) += t22;
            hess_xx(ri, x1i) += t2;
            hess_xx(ri, y1i) += t1;
            hess_xx(ri, ri) += (-0.4e1 * t8 * t15 * t12 * t18 + t26 *
                                                                (t16 * t5 * t14 * t6 * t7 * pow(t8, 0.2e1) * t11 * t18 -
                                                                 0.4e1 * t9 *
                                                                 (-t5 * t7 * t10 * t19 + t15 * t12 * t21))) * r + t17 -
                               t19;
        }
    }
}

#include "../../include/Energy/CellFunctionPerimeter.h"
#include <iostream>

void CellFunctionPerimeter::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                     const VectorXi &btype, double &value,
                                     const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        if (btype(i) == 0) {
            value += sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + epsilon);
        } else {
            value += 0.2e1 * asin(sqrt(x0 * x0 - 0.2e1 * x1 * x0 + x1 * x1 + pow(-y1 + y0, 0.2e1)) / q0 / 0.2e1) * q0;
        }
    }
}

void CellFunctionPerimeter::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                        const VectorXi &btype,
                                        VectorXT &gradient_c,
                                        VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        if (btype(i) == 0) {
            t1 = x1 - x0;
            t2 = -y1 + y0;
            t3 = pow(t1, 0.2e1) + pow(t2, 0.2e1) + epsilon;
            t3 = pow(t3, -0.1e1 / 0.2e1);
            t2 = t3 * t2;
            t1 = t3 * t1;

            gradient_x(x0i) += -t1;
            gradient_x(y0i) += t2;
            gradient_x(x1i) += t1;
            gradient_x(y1i) += -t2;
        } else {
            t1 = -y1 + y0;
            t2 = pow(t1, 0.2e1) + x0 * x0 - 0.2e1 * x1 * x0 + x1 * x1;
            t3 = pow(t2, -0.1e1 / 0.2e1);
            t4 = 0.1e1 / q0;
            t5 = -t2 * pow(t4, 0.2e1) + 0.4e1;
            t5 = pow(t5, -0.1e1 / 0.2e1);
            t2 = t2 * t3;
            t3 = 0.2e1 * t3;
            t6 = t3 * (x1 - x0) * t5;
            t1 = t3 * t1 * t5;
            gradient_x(x0i) += -t6;
            gradient_x(y0i) += t1;
            gradient_x(x1i) += t6;
            gradient_x(y1i) += -t1;
            gradient_x(q0i) += -0.2e1 * t2 * t4 * t5 + 0.2e1 * asin(t2 * t4 / 0.2e1);
        }
    }
}

void
CellFunctionPerimeter::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                  const VectorXi &btype, MatrixXT &hessian,
                                  const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / nx;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, q0, x1, y1, q1;
    int x0i, y0i, q0i, x1i, y1i, q1i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * nx + 0;
        y0i = i * nx + 1;
        q0i = i * nx + 2;
        x1i = next(i) * nx + 0;
        y1i = next(i) * nx + 1;
        q1i = next(i) * nx + 2;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        q0 = nodes(q0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);
        q1 = nodes(q1i);

        if (btype(i) == 0) {
            t1 = x1 - x0;
            t2 = -y1 + y0;
            t3 = pow(t1, 0.2e1);
            t4 = pow(t2, 0.2e1);
            t5 = t4 + t3 + epsilon;
            t6 = pow(t5, -0.3e1 / 0.2e1);
            t5 = t5 * t6;
            t3 = -t3 * t6 + t5;
            t1 = t6 * t1 * t2;
            t2 = -t4 * t6 + t5;

            hess_xx(x0i, x0i) += t3;
            hess_xx(x0i, y0i) += t1;
            hess_xx(x0i, x1i) += -t3;
            hess_xx(x0i, y1i) += -t1;
            hess_xx(y0i, x0i) += t1;
            hess_xx(y0i, y0i) += t2;
            hess_xx(y0i, x1i) += -t1;
            hess_xx(y0i, y1i) += -t2;
            hess_xx(x1i, x0i) += -t3;
            hess_xx(x1i, y0i) += -t1;
            hess_xx(x1i, x1i) += t3;
            hess_xx(x1i, y1i) += t1;
            hess_xx(y1i, x0i) += -t1;
            hess_xx(y1i, y0i) += -t2;
            hess_xx(y1i, x1i) += t1;
            hess_xx(y1i, y1i) += t2;
        } else {
            t1 = -y1 + y0;
            t2 = pow(t1, 0.2e1);
            t3 = x0 * x0 - 0.2e1 * x1 * x0 + x1 * x1 + t2;
            t4 = pow(t3, -0.3e1 / 0.2e1);
            t5 = x1 - x0;
            t6 = 0.1e1 / q0;
            t7 = pow(t6, 0.2e1);
            t6 = t6 * t7;
            t8 = -t3 * t7 + 0.4e1;
            t9 = pow(t8, -0.3e1 / 0.2e1);
            t8 = t8 * t9;
            t10 = t3 * t4;
            t11 = pow(t5, 0.2e1);
            t12 = t4 * t8;
            t13 = pow(t3, 0.2e1);
            t14 = 0.2e1 * t13 * t4;
            t15 = t14 * t5 * t9 * t6;
            t11 = 0.2e1 * t10 * (-t11 * t9 * t7 - t8) + 0.2e1 * t12 * t11;
            t5 = 0.2e1 * t5 * t1 * (t10 * t9 * t7 - t12);
            t1 = t14 * t1 * t9 * t6;
            t2 = 0.2e1 * t10 * (t2 * t9 * t7 + t8) - 0.2e1 * t12 * t2;
            hess_xx(x0i, x0i) += -t11;
            hess_xx(x0i, y0i) += -t5;
            hess_xx(x0i, x1i) += t11;
            hess_xx(x0i, y1i) += t5;
            hess_xx(x0i, q0i) += t15;
            hess_xx(y0i, x0i) += -t5;
            hess_xx(y0i, y0i) += t2;
            hess_xx(y0i, x1i) += t5;
            hess_xx(y0i, y1i) += -t2;
            hess_xx(y0i, q0i) += -t1;
            hess_xx(x1i, x0i) += t11;
            hess_xx(x1i, y0i) += t5;
            hess_xx(x1i, x1i) += -t11;
            hess_xx(x1i, y1i) += -t5;
            hess_xx(x1i, q0i) += -t15;
            hess_xx(y1i, x0i) += t5;
            hess_xx(y1i, y0i) += -t2;
            hess_xx(y1i, x1i) += -t5;
            hess_xx(y1i, y1i) += t2;
            hess_xx(y1i, q0i) += t1;
            hess_xx(q0i, x0i) += t15;
            hess_xx(q0i, y0i) += -t1;
            hess_xx(q0i, x1i) += -t15;
            hess_xx(q0i, y1i) += t1;
            hess_xx(q0i, q0i) += 0.2e1 * t3 * t13 * t4 * pow(t7, 0.2e1) * t9;
        }
    }
}

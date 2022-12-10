#include "../../include/Energy/CellFunctionPerimeter.h"
#include <iostream>

void CellFunctionPerimeter::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                     const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = ((i + 1) % n_nodes) * 2 + 0;
        y1i = ((i + 1) % n_nodes) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        value += sqrt((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0) + epsilon);
    }
}

void CellFunctionPerimeter::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                        VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2, t3;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = ((i + 1) % n_nodes) * 2 + 0;
        y1i = ((i + 1) % n_nodes) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

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
    }
}

void CellFunctionPerimeter::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
                                       const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2, t3, t4, t5, t6;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = ((i + 1) % n_nodes) * 2 + 0;
        y1i = ((i + 1) % n_nodes) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

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
    }
}

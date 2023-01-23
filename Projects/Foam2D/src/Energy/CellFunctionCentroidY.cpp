#include "../../include/Energy/CellFunctionCentroidY.h"
#include <iostream>

void
CellFunctionCentroidY::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                                const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    for (int i = 0; i < n_nodes; i++) {
        x0 = nodes(i * 2 + 0);
        y0 = nodes(i * 2 + 1);
        x1 = nodes(next(i) * 2 + 0);
        y1 = nodes(next(i) * 2 + 1);

        value += (x0 * y1 - x1 * y0) * (y0 + y1) / 6.0;
    }
}

void CellFunctionCentroidY::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                        VectorXT &gradient_c,
                                        VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        t1 = y0 + y1;
        t2 = 0.1e1 / 0.6e1;
        gradient_x(x0i) += t2 * y1 * t1;
        gradient_x(y0i) += -t2 * (-x0 * y1 + (y0 + t1) * x1);
        gradient_x(x1i) += -t2 * y0 * t1;
        gradient_x(y1i) += t2 * ((y1 + t1) * x0 - x1 * y0);
    }
}

void
CellFunctionCentroidY::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
                                  const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2, t3, t4, t5;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        t1 = y0 / 0.6e1;
        t2 = y1 / 0.3e1 + t1;
        t3 = y1 / 0.6e1;
        t4 = -y0 / 0.3e1 - t3;
        t5 = -x1 / 0.6e1 + x0 / 0.6e1;
        hess_xx(x0i, x0i) += 0;
        hess_xx(x0i, y0i) += t3;
        hess_xx(x0i, x1i) += 0;
        hess_xx(x0i, y1i) += t2;
        hess_xx(y0i, x0i) += t3;
        hess_xx(y0i, y0i) += -x1 / 0.3e1;
        hess_xx(y0i, x1i) += t4;
        hess_xx(y0i, y1i) += t5;
        hess_xx(x1i, x0i) += 0;
        hess_xx(x1i, y0i) += t4;
        hess_xx(x1i, x1i) += 0;
        hess_xx(x1i, y1i) += -t1;
        hess_xx(y1i, x0i) += t2;
        hess_xx(y1i, y0i) += t5;
        hess_xx(y1i, x1i) += -t1;
        hess_xx(y1i, y1i) += x0 / 0.3e1;
    }
}

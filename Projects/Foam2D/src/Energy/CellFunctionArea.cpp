#include "../../include/Energy/CellFunctionArea.h"
#include <iostream>

void
CellFunctionArea::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                           const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    for (int i = 0; i < n_nodes; i++) {
        x0 = nodes(i * 2 + 0);
        y0 = nodes(i * 2 + 1);
        x1 = nodes(next(i) * 2 + 0);
        y1 = nodes(next(i) * 2 + 1);

        value += 0.5 * (x0 * y1 - x1 * y0);
    }
}

void
CellFunctionArea::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_c,
                              VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        gradient_x(x0i) += 0.5 * y1;
        gradient_x(y0i) += -0.5 * x1;
        gradient_x(x1i) += -0.5 * y0;
        gradient_x(y1i) += 0.5 * x0;
    }
}

void
CellFunctionArea::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
                             const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        hess_xx(x0i, y1i) += 0.5;
        hess_xx(y0i, x1i) += -0.5;
        hess_xx(x1i, y0i) += -0.5;
        hess_xx(y1i, x0i) += 0.5;
    }
}

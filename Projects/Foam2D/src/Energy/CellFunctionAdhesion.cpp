#include "../../include/Energy/CellFunctionAdhesion.h"
#include <iostream>

void CellFunctionAdhesion::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                                    const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    for (int i = 0; i < n_nodes; i++) {
        double a = cellInfo->neighbor_affinity(i);

        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        value += a * ((x1 - x0) * (x1 - x0) + (y1 - y0) * (y1 - y0));
    }
}

void CellFunctionAdhesion::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                       VectorXT &gradient_c,
                                       VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2, t3;
    for (int i = 0; i < n_nodes; i++) {
        double a = cellInfo->neighbor_affinity(i);

        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        t1 = 0.2e1 * a;
        t2 = t1 * (x1 - x0);
        t1 = t1 * (-y1 + y0);

        gradient_x(x0i) += -t2;
        gradient_x(y0i) += t1;
        gradient_x(x1i) += t2;
        gradient_x(y1i) += -t1;
    }
}

void
CellFunctionAdhesion::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
                                 const CellInfo *cellInfo) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    double x0, y0, x1, y1;
    int x0i, y0i, x1i, y1i;
    double t1, t2, t3, t4, t5, t6;
    for (int i = 0; i < n_nodes; i++) {
        double a = cellInfo->neighbor_affinity(i);

        x0i = i * 2 + 0;
        y0i = i * 2 + 1;
        x1i = next(i) * 2 + 0;
        y1i = next(i) * 2 + 1;

        x0 = nodes(x0i);
        y0 = nodes(y0i);
        x1 = nodes(x1i);
        y1 = nodes(y1i);

        t1 = 0.2e1 * a;

        hess_xx(x0i, x0i) += t1;
        hess_xx(x0i, y0i) += 0;
        hess_xx(x0i, x1i) += -t1;
        hess_xx(x0i, y1i) += 0;
        hess_xx(y0i, x0i) += 0;
        hess_xx(y0i, y0i) += t1;
        hess_xx(y0i, x1i) += 0;
        hess_xx(y0i, y1i) += -t1;
        hess_xx(x1i, x0i) += -t1;
        hess_xx(x1i, y0i) += 0;
        hess_xx(x1i, x1i) += t1;
        hess_xx(x1i, y1i) += 0;
        hess_xx(y1i, x0i) += 0;
        hess_xx(y1i, y0i) += -t1;
        hess_xx(y1i, x1i) += 0;
        hess_xx(y1i, y1i) += t1;
    }
}

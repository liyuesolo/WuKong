#include "../../include/Energy/CellFunctionArea.h"
#include <iostream>

void CellFunctionArea::addValue(const VectorXT &site, const VectorXT &nodes, double &value) const {
    int n_nodes = nodes.rows() / 2;

    double x0 = nodes(0);
    double y0 = nodes(1);
    double x1, y1, x2, y2;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1 = nodes(i * 2 + 0);
        y1 = nodes(i * 2 + 1);
        x2 = nodes((i + 1) * 2 + 0);
        y2 = nodes((i + 1) * 2 + 1);

        value += 0.5 * ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0));
    }
}

void CellFunctionArea::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                   VectorXT &gradient_x) const {
    int n_nodes = nodes.rows() / 2;

    int x0i = 0, y0i = 1;
    double x0 = nodes(x0i);
    double y0 = nodes(y0i);
    double x1, y1, x2, y2;
    int x1i, y1i, x2i, y2i;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1i = i * 2 + 0;
        y1i = i * 2 + 1;
        x2i = (i + 1) * 2 + 0;
        y2i = (i + 1) * 2 + 1;

        x1 = nodes(x1i);
        y1 = nodes(y1i);
        x2 = nodes(x2i);
        y2 = nodes(y2i);

        gradient_x(x0i) += -0.5 * (y2 - y1);
        gradient_x(y0i) += 0.5 * (x2 - x1);
        gradient_x(x1i) += -0.5 * (-y2 + y0);
        gradient_x(y1i) += 0.5 * (-x2 + x0);
        gradient_x(x2i) += 0.5 * (-y1 + y0);
        gradient_x(y2i) += -0.5 * (-x1 + x0);
    }
}

void CellFunctionArea::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    int x0i = 0, y0i = 1;
    int x1i, y1i, x2i, y2i;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1i = i * 2 + 0;
        y1i = i * 2 + 1;
        x2i = (i + 1) * 2 + 0;
        y2i = (i + 1) * 2 + 1;

        hess_xx(x0i, y1i) += 0.5;
        hess_xx(x0i, y2i) += -0.5;
        hess_xx(y0i, x1i) += -0.5;
        hess_xx(y0i, x2i) += 0.5;
        hess_xx(x1i, y2i) += 0.5;
        hess_xx(x1i, y0i) += -0.5;
        hess_xx(y1i, x2i) += -0.5;
        hess_xx(y1i, x0i) += 0.5;
        hess_xx(x2i, y0i) += 0.5;
        hess_xx(x2i, y1i) += -0.5;
        hess_xx(y2i, x0i) += -0.5;
        hess_xx(y2i, x1i) += 0.5;
    }
}

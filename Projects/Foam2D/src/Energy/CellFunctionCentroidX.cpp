#include "../../include/Energy/CellFunctionCentroidX.h"
#include <iostream>

void CellFunctionCentroidX::addValue(const VectorXT &site, const VectorXT &nodes, double &value) const {
    int n_nodes = nodes.rows() / 2;

    double x0 = nodes(0);
    double y0 = nodes(1);
    double x1, y1, x2, y2;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1 = nodes(i * 2 + 0);
        y1 = nodes(i * 2 + 1);
        x2 = nodes((i + 1) * 2 + 0);
        y2 = nodes((i + 1) * 2 + 1);

        value += ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)) * (x0 + x1 + x2) / 6.0;
    }
}

void CellFunctionCentroidX::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                        VectorXT &gradient_x) const {
    int n_nodes = nodes.rows() / 2;

    int x0i = 0, y0i = 1;
    double x0 = nodes(x0i);
    double y0 = nodes(y0i);
    double x1, y1, x2, y2;
    int x1i, y1i, x2i, y2i;
    double t1, t2, t3, t4, t5, t6, t7, t8;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1i = i * 2 + 0;
        y1i = i * 2 + 1;
        x2i = (i + 1) * 2 + 0;
        y2i = (i + 1) * 2 + 1;

        x1 = nodes(x1i);
        y1 = nodes(y1i);
        x2 = nodes(x2i);
        y2 = nodes(y2i);

        t1 = x0 + x1 + x2;
        t2 = -x1 + x0;
        t3 = -y2 + y0;
        t4 = x2 - x0;
        t5 = -y1 + y0;
        t6 = t4 * t5;
        t7 = t2 * t3;
        t8 = 0.1e1 / 0.6e1;
        gradient_x(x0i) += t8 * ((y1 - y2) * t1 + t6 + t7);
        gradient_x(y0i) += t8 * (x2 - x1) * t1;
        gradient_x(x1i) += t8 * (t3 * (-t1 + t2) + t6);
        gradient_x(y1i) += -t8 * t4 * t1;
        gradient_x(x2i) += t8 * (t5 * (t1 + t4) + t7);
        gradient_x(y2i) += -t8 * t2 * t1;
    }
}

void CellFunctionCentroidX::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const {
    int n_nodes = nodes.rows() / 2;

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());
//    MatrixXT hess_xx = MatrixXT::Zero(nodes.rows(), nodes.rows());

    int x0i = 0, y0i = 1;
    double x0 = nodes(x0i);
    double y0 = nodes(y0i);
    double x1, y1, x2, y2;
    int x1i, y1i, x2i, y2i;
    double t1, t2, t3, t4, t5, t6, t7, t8, t9, t10, t11, t12, t13, t14, t15, t16;
    for (int i = 1; i < n_nodes - 1; i++) {
        x1i = i * 2 + 0;
        y1i = i * 2 + 1;
        x2i = (i + 1) * 2 + 0;
        y2i = (i + 1) * 2 + 1;

        x1 = nodes(x1i);
        y1 = nodes(y1i);
        x2 = nodes(x2i);
        y2 = nodes(y2i);

        t1 = y1 - y2;
        t2 = -y1 + y0;
        t3 = 0.1e1 / 0.6e1;
        t4 = x0 / 0.3e1;
        t5 = t3 * x1;
        t6 = t4 + t5;
        t7 = -y2 + y0;
        t8 = t3 * x2;
        t4 = -t4 - t8;
        t9 = t3 * (x2 - x1);
        t10 = t3 * t2;
        t11 = t3 * t7;
        t12 = t3 * x0;
        t13 = x1 / 0.3e1;
        t14 = -t12 - t13;
        t15 = x2 / 0.3e1;
        t12 = t15 + t12;
        t8 = t8 + t13;
        t13 = t3 * t1;
        t16 = t3 * (x2 - x0);
        t5 = -t15 - t5;
        t3 = t3 * (-x1 + x0);
        hess_xx(x0i, x0i) += t1 / 0.3e1;
        hess_xx(x0i, y0i) += t9;
        hess_xx(x0i, x1i) += -t10;
        hess_xx(x0i, y1i) += t6;
        hess_xx(x0i, x2i) += t11;
        hess_xx(x0i, y2i) += t4;
        hess_xx(y0i, x0i) += t9;
        hess_xx(y0i, y0i) += 0;
        hess_xx(y0i, x1i) += t14;
        hess_xx(y0i, y1i) += 0;
        hess_xx(y0i, x2i) += t12;
        hess_xx(y0i, y2i) += 0;
        hess_xx(x1i, x0i) += -t10;
        hess_xx(x1i, y0i) += t14;
        hess_xx(x1i, x1i) += -t7 / 0.3e1;
        hess_xx(x1i, y1i) += -t16;
        hess_xx(x1i, x2i) += -t13;
        hess_xx(x1i, y2i) += t8;
        hess_xx(y1i, x0i) += t6;
        hess_xx(y1i, y0i) += 0;
        hess_xx(y1i, x1i) += -t16;
        hess_xx(y1i, y1i) += 0;
        hess_xx(y1i, x2i) += t5;
        hess_xx(y1i, y2i) += 0;
        hess_xx(x2i, x0i) += t11;
        hess_xx(x2i, y0i) += t12;
        hess_xx(x2i, x1i) += -t13;
        hess_xx(x2i, y1i) += t5;
        hess_xx(x2i, x2i) += t2 / 0.3e1;
        hess_xx(x2i, y2i) += -t3;
        hess_xx(y2i, x0i) += t4;
        hess_xx(y2i, y0i) += 0;
        hess_xx(y2i, x1i) += t8;
        hess_xx(y2i, y1i) += 0;
        hess_xx(y2i, x2i) += -t3;
        hess_xx(y2i, y2i) += 0;
    }

//    VectorXT temp;
//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, temp, grad);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, temp, gradp);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "centroidx hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " "
//                      << hess_xx(j, i) << std::endl;
//        }
//    }
}

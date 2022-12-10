#include "../../include/Energy/CellFunctionAreaBarrier.h"
#include <iostream>

void CellFunctionAreaBarrier::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                       const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    value += epsilon * pow(area, exponent);
}

void CellFunctionAreaBarrier::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                          VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(gradient_x.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);

    gradient_x += epsilon * (exponent * pow(area, exponent - 1.0) * area_gradient_x);
}

void CellFunctionAreaBarrier::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
                                         const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);

    MatrixXT area_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    area_function.addHessian(site, nodes, area_hessian, cellInfo);

    Eigen::Ref<MatrixXT> area_hess = area_hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    Eigen::Ref<MatrixXT> hess = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    hess += epsilon *
            (exponent * (exponent - 1.0) * pow(area, exponent - 2.0) * area_gradient_x * area_gradient_x.transpose() +
             exponent * pow(area, exponent - 1.0) * area_hess);

//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    addGradient(site, nodes, temp, grad);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, temp, gradp);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "at hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " " << hessian(site.rows() + j, site.rows() +  i) << std::endl;
//        }
//    }
}

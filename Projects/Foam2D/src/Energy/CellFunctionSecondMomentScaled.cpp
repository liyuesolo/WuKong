#include "../../include/Energy/CellFunctionSecondMomentScaled.h"
#include <iostream>

void
CellFunctionSecondMomentScaled::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                         const VectorXi &btype, double &value,
                                         const CellInfo *cellInfo) const {
    double moment = 0;
    moment_function.addValue(site, nodes, next, btype, moment, cellInfo);

    double scale = 1.0 / pow(cellInfo->target_area, 2.0);
    value += scale * moment;
}

void CellFunctionSecondMomentScaled::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                                 const VectorXi &btype,
                                                 VectorXT &gradient_c,
                                                 VectorXT &gradient_x, const CellInfo *cellInfo) const {

    VectorXT temp;
    VectorXT moment_gradient_x = VectorXT::Zero(gradient_x.rows());
    moment_function.addGradient(site, nodes, next, btype, temp, moment_gradient_x, cellInfo);

    double scale = 1.0 / pow(cellInfo->target_area, 2.0);
    gradient_x += scale * moment_gradient_x;
}

void
CellFunctionSecondMomentScaled::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                           const VectorXi &btype, MatrixXT &hessian,
                                           const CellInfo *cellInfo) const {

    MatrixXT moment_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    moment_function.addHessian(site, nodes, next, btype, moment_hessian, cellInfo);

    double scale = 1.0 / pow(cellInfo->target_area, 2.0);
    hessian += scale * moment_hessian;

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

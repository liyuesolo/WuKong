#include "../../include/Energy/CellFunctionAreaTarget.h"
#include <iostream>

void
CellFunctionAreaTarget::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                 const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double target_reciprocal = 1.0 / cellInfo->target_area;
    value += ((area * target_reciprocal - 1) * (area * target_reciprocal - 1));
}

void CellFunctionAreaTarget::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                         VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(gradient_x.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);

    double target_reciprocal = 1.0 / cellInfo->target_area;
    gradient_x += (2 * (area * target_reciprocal - 1) * target_reciprocal * area_gradient_x);
}

void CellFunctionAreaTarget::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
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

    double target_reciprocal = 1.0 / cellInfo->target_area;
    hess += (2 * area_gradient_x * area_gradient_x.transpose() * target_reciprocal * target_reciprocal +
             2 * (area * target_reciprocal - 1) * area_hess * target_reciprocal);

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

void CellFunctionAreaTarget::addGradient_tau(const VectorXT &site, const VectorXT &nodes, double &gradient_tau, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);

    double tau = 1.0 / cellInfo->target_area;
    gradient_tau += 2 * area * (area * tau - 1);
}

void CellFunctionAreaTarget::addHessian_tau(const VectorXT &site, const VectorXT &nodes, double &hessian_tau_tau,
                                            VectorXT &hessian_tau_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);

    double tau = 1.0 / cellInfo->target_area;
    hessian_tau_tau += 2 * area * area;
    hessian_tau_x += (4 * area * tau - 2) * area_gradient_x;
}

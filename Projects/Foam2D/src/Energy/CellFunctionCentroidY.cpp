#include "../../include/Energy/CellFunctionCentroidY.h"
#include <iostream>

void
CellFunctionCentroidY::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                const VectorXi &btype, double &value,
                                const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double weighted_mean = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, weighted_mean, cellInfo);

    value += weighted_mean / area;
}

void CellFunctionCentroidY::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                        const VectorXi &btype,
                                        VectorXT &gradient_c,
                                        VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double weighted_mean = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, weighted_mean, cellInfo);

    VectorXT temp;
    VectorXT area_gradient = VectorXT::Zero(gradient_x.rows());
    area_function.addGradient(site, nodes, next, btype, temp, area_gradient, cellInfo);
    VectorXT weighted_mean_gradient = VectorXT::Zero(nodes.rows());
    weighted_mean_function.addGradient(site, nodes, next, btype, temp, weighted_mean_gradient, cellInfo);

    gradient_x += weighted_mean_gradient / area - weighted_mean * area_gradient / pow(area, 2);
}

void CellFunctionCentroidY::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                       const VectorXi &btype,
                                       MatrixXT &hessian,
                                       const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double weighted_mean = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, weighted_mean, cellInfo);

    VectorXT temp;
    VectorXT area_gradient = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, next, btype, temp, area_gradient, cellInfo);
    VectorXT weighted_mean_gradient = VectorXT::Zero(nodes.rows());
    weighted_mean_function.addGradient(site, nodes, next, btype, temp, weighted_mean_gradient, cellInfo);

    MatrixXT area_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    area_function.addHessian(site, nodes, next, btype, area_hessian, cellInfo);
    MatrixXT weighted_mean_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    weighted_mean_function.addHessian(site, nodes, next, btype, weighted_mean_hessian, cellInfo);

    Eigen::Ref<MatrixXT> hess_xx = hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    Eigen::Ref<MatrixXT> hess_area_xx = area_hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    Eigen::Ref<MatrixXT> hess_wm_xx = weighted_mean_hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    hess_xx += hess_wm_xx / area
               - area_gradient * weighted_mean_gradient.transpose() / pow(area, 2.0)
               - weighted_mean_gradient * area_gradient.transpose() / pow(area, 2.0)
               + 2 * weighted_mean * area_gradient * area_gradient.transpose() / pow(area, 3.0)
               - weighted_mean * hess_area_xx / pow(area, 2.0);
}

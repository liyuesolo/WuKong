#include "../../include/Energy/CellFunctionCentroidYTarget.h"
#include <iostream>

void
CellFunctionCentroidYTarget::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                      const VectorXi &btype, double &value,
                                      const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double centroid = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, centroid, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    value += multiplier * pow(site(1) - centroid / area, 2.0);
}

void CellFunctionCentroidYTarget::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                              const VectorXi &btype,
                                              VectorXT &gradient_c,
                                              VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double centroid = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, centroid, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(gradient_x.rows());
    area_function.addGradient(site, nodes, next, btype, temp, area_gradient_x, cellInfo);
    VectorXT centroid_gradient_x = VectorXT::Zero(nodes.rows());
    weighted_mean_function.addGradient(site, nodes, next, btype, temp, centroid_gradient_x, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    gradient_c(1) += multiplier * 2 * (site(1) - centroid / area);
    gradient_x += multiplier * 2 * (site(1) - centroid / area) *
                  (centroid * area_gradient_x / pow(area, 2.0) - centroid_gradient_x / area);

//    VectorXT this_grad_c = VectorXT::Zero(site.rows());
//    this_grad_c(1) = 2 * (site(1) - centroid / area);
//    VectorXT this_grad_x = 2 * (site(1) - centroid / area) *
//                           (centroid * area_gradient_x / pow(area, 2.0) - centroid_gradient_x / area);
//    double f = 0;
//    addValue(site, nodes, f);
//    double eps = 1e-6;
//    for (int i = 0; i < site.rows(); i++) {
//        VectorXT xp = site;
//        xp(i) += eps;
//        double fp = 0;
//        addValue(xp, nodes, fp);
//        std::cout << "centroidytarget  grad c[" << i << "] " << (fp - f) / eps << " "
//                  << this_grad_c(i) << std::endl;
//    }
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        double fp = 0;
//        addValue(site, xp, fp);
//        std::cout << "centroidytarget  grad[" << i << "] " << (fp - f) / eps << " "
//                  << this_grad_x(i) << std::endl;
//    }
}

void CellFunctionCentroidYTarget::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                             const VectorXi &btype,
                                             MatrixXT &hessian,
                                             const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, next, btype, area, cellInfo);
    double centroid = 0;
    weighted_mean_function.addValue(site, nodes, next, btype, centroid, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, next, btype, temp, area_gradient_x, cellInfo);
    VectorXT centroid_gradient_x = VectorXT::Zero(nodes.rows());
    weighted_mean_function.addGradient(site, nodes, next, btype, temp, centroid_gradient_x, cellInfo);

    VectorXT area_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    area_gradient.segment(site.rows(), nodes.rows()) = area_gradient_x;
    VectorXT centroid_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    centroid_gradient.segment(site.rows(), nodes.rows()) = centroid_gradient_x;
    VectorXT siteY_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    siteY_gradient(1) = 1.0;

    MatrixXT area_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    area_function.addHessian(site, nodes, next, btype, area_hessian, cellInfo);
    MatrixXT centroid_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    weighted_mean_function.addHessian(site, nodes, next, btype, centroid_hessian, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    VectorXT aaa = siteY_gradient - centroid_gradient / area + centroid * area_gradient / pow(area, 2.0);
    hessian += multiplier * 2 * aaa * aaa.transpose();
    hessian += multiplier * 2 * (site(1) - centroid / area) * (
            area_gradient * centroid_gradient.transpose() / pow(area, 2.0)
            + centroid_gradient * area_gradient.transpose() / pow(area, 2.0)
            - centroid_hessian / area
            + centroid * area_hessian / pow(area, 2.0)
            - 2 * centroid * area_gradient * area_gradient.transpose() / pow(area, 3.0));

//    MatrixXT this_hess = 0 * hessian;
//    this_hess += 2 * aaa * aaa.transpose();
//    this_hess += 2 * (site(1) - centroid / area) * (
//            area_gradient * centroid_gradient.transpose() / pow(area, 2.0)
//            + centroid_gradient * area_gradient.transpose() / pow(area, 2.0)
//            - centroid_hessian / area
//            + centroid * area_hessian / pow(area, 2.0)
//            - 2 * centroid * area_gradient * area_gradient.transpose() / pow(area, 3.0));
//    VectorXT grad = VectorXT::Zero(nodes.rows());
//    VectorXT gradc = VectorXT::Zero(site.rows());
//    addGradient(site, nodes, gradc, grad);
//    double eps = 1e-6;
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        VectorXT gradp = VectorXT::Zero(nodes.rows());
//        addGradient(site, xp, gradc, gradp);
//        for (int j = 0; j < nodes.rows(); j++) {
//            std::cout << "centroidytarget  hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " "
//                      << this_hess(site.rows() + j, site.rows() + i) << std::endl;
//        }
//    }
}

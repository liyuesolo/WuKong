#include "../../include/Energy/CellFunctionCentroidXTarget.h"
#include <iostream>

void CellFunctionCentroidXTarget::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                           const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double centroid = 0;
    centroid_function.addValue(site, nodes, centroid, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    value += multiplier * pow(site(0) - centroid / area, 2.0);
}

void CellFunctionCentroidXTarget::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                              VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double centroid = 0;
    centroid_function.addValue(site, nodes, centroid, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);
    VectorXT centroid_gradient_x = VectorXT::Zero(nodes.rows());
    centroid_function.addGradient(site, nodes, temp, centroid_gradient_x, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    gradient_c(0) += multiplier * 2 * (site(0) - centroid / area);
    gradient_x += multiplier * 2 * (site(0) - centroid / area) *
                  (centroid * area_gradient_x / pow(area, 2.0) - centroid_gradient_x / area);

//    VectorXT this_grad_c = VectorXT::Zero(site.rows());
//    this_grad_c(0) = 2 * (site(0) - centroid / area);
//    VectorXT this_grad_x = 2 * (site(0) - centroid / area) *
//                           (centroid * area_gradient_x / pow(area, 2.0) - centroid_gradient_x / area);
//    double f = 0;
//    addValue(site, nodes, f);
//    double eps = 1e-6;
//    for (int i = 0; i < site.rows(); i++) {
//        VectorXT xp = site;
//        xp(i) += eps;
//        double fp = 0;
//        addValue(xp, nodes, fp);
//        std::cout << "centroidxtarget  grad c[" << i << "] " << (fp - f) / eps << " "
//                  << this_grad_c(i) << std::endl;
//    }
//    for (int i = 0; i < nodes.rows(); i++) {
//        VectorXT xp = nodes;
//        xp(i) += eps;
//        double fp = 0;
//        addValue(site, xp, fp);
//        std::cout << "centroidxtarget  grad[" << i << "] " << (fp - f) / eps << " "
//                  << this_grad_x(i) << std::endl;
//    }
}

void CellFunctionCentroidXTarget::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
                                             const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double centroid = 0;
    centroid_function.addValue(site, nodes, centroid, cellInfo);

    VectorXT temp;
    VectorXT area_gradient_x = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, temp, area_gradient_x, cellInfo);
    VectorXT centroid_gradient_x = VectorXT::Zero(nodes.rows());
    centroid_function.addGradient(site, nodes, temp, centroid_gradient_x, cellInfo);

    VectorXT area_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    area_gradient.segment(site.rows(), nodes.rows()) = area_gradient_x;
    VectorXT centroid_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    centroid_gradient.segment(site.rows(), nodes.rows()) = centroid_gradient_x;
    VectorXT siteX_gradient = VectorXT::Zero(site.rows() + nodes.rows());
    siteX_gradient(0) = 1.0;

    MatrixXT area_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    area_function.addHessian(site, nodes, area_hessian, cellInfo);
    MatrixXT centroid_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    centroid_function.addHessian(site, nodes, centroid_hessian, cellInfo);

    double multiplier = cellInfo->agent ? 20 : 1;
    VectorXT aaa = siteX_gradient - centroid_gradient / area + centroid * area_gradient / pow(area, 2.0);
    hessian += multiplier * 2 * aaa * aaa.transpose();
    hessian += multiplier * 2 * (site(0) - centroid / area) * (
            area_gradient * centroid_gradient.transpose() / pow(area, 2.0)
            + centroid_gradient * area_gradient.transpose() / pow(area, 2.0)
            - centroid_hessian / area
            + centroid * area_hessian / pow(area, 2.0)
            - 2 * centroid * area_gradient * area_gradient.transpose() / pow(area, 3.0));

//    MatrixXT this_hess = 0 * hessian;
//    this_hess += 2 * aaa * aaa.transpose();
//    this_hess += 2 * (site(0) - centroid / area) * (
//            area_gradient * centroid_gradient.transpose() / pow(area, 2.0)
//            + centroid_gradient * area_gradient.transpose() / pow(area, 2.0)
//            - centroid_hessian / area
//            + centroid * area_hessian / pow(area, 2.0)
//            - 2 * centroid * area_gradient * area_gradient.transpose() / pow(area, 3.0));
//    VectorXT gradx = VectorXT::Zero(nodes.rows());
//    VectorXT gradc = VectorXT::Zero(site.rows());
//    addGradient(site, nodes, gradc, gradx);
//    VectorXT grad(site.rows() + nodes.rows());
//    grad << gradc, gradx;
//    double eps = 1e-6;
//    for (int i = 0; i < hessian.rows(); i++) {
//        VectorXT sitep = site;
//        VectorXT nodesp = nodes;
//        if (i < site.rows()) {
//            sitep(i) += eps;
//        } else {
//            nodesp(i - site.rows()) += eps;
//        }
//        VectorXT gradxp = VectorXT::Zero(nodes.rows());
//        VectorXT gradcp = VectorXT::Zero(site.rows());
//        addGradient(sitep, nodesp, gradcp, gradxp);
//        VectorXT gradp(site.rows() + nodes.rows());
//        gradp << gradcp, gradxp;
//        for (int j = 0; j < hessian.rows(); j++) {
//            std::cout << "centroidxtarget  hess[" << j << "," << i << "] " << (gradp[j] - grad[j]) / eps << " "
//                      << this_hess(j, i) << std::endl;
//        }
//    }
}

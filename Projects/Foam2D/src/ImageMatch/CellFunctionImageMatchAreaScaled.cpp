#include "../../include/ImageMatch/CellFunctionImageMatchAreaScaled.h"
#include <iostream>

void CellFunctionImageMatchAreaScaled::addValue(const VectorXT &site, const VectorXT &nodes, double &value,
                                          const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double image_match = 0;
    image_match_function.addValue(site, nodes, image_match, cellInfo);

    value += image_match / area;
}

void CellFunctionImageMatchAreaScaled::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                             VectorXT &gradient_x, const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double image_match = 0;
    image_match_function.addValue(site, nodes, image_match, cellInfo);

    VectorXT temp;
    VectorXT area_gradient = VectorXT::Zero(gradient_x.rows());
    area_function.addGradient(site, nodes, temp, area_gradient, cellInfo);
    VectorXT image_match_gradient = VectorXT::Zero(nodes.rows());
    image_match_function.addGradient(site, nodes, temp, image_match_gradient, cellInfo);

    gradient_x += image_match_gradient / area - image_match * area_gradient / pow(area, 2);
}

void CellFunctionImageMatchAreaScaled::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian,
                                            const CellInfo *cellInfo) const {
    double area = 0;
    area_function.addValue(site, nodes, area, cellInfo);
    double image_match = 0;
    image_match_function.addValue(site, nodes, image_match, cellInfo);

    VectorXT temp;
    VectorXT area_gradient = VectorXT::Zero(nodes.rows());
    area_function.addGradient(site, nodes, temp, area_gradient, cellInfo);
    VectorXT image_match_gradient = VectorXT::Zero(nodes.rows());
    image_match_function.addGradient(site, nodes, temp, image_match_gradient, cellInfo);

    MatrixXT area_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    area_function.addHessian(site, nodes, area_hessian, cellInfo);
    MatrixXT image_match_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
    image_match_function.addHessian(site, nodes, image_match_hessian, cellInfo);

    Eigen::Ref<MatrixXT> area_hess = area_hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    Eigen::Ref<MatrixXT> image_match_hess = image_match_hessian.bottomRightCorner(nodes.rows(), nodes.rows());
    Eigen::Ref<MatrixXT> hess = hessian.bottomRightCorner(nodes.rows(), nodes.rows());

    hess += image_match_hess / area
            - area_gradient * image_match_gradient.transpose() / pow(area, 2.0)
            - image_match_gradient * area_gradient.transpose() / pow(area, 2.0)
            + 2 * image_match * area_gradient * area_gradient.transpose() / pow(area, 3.0)
            - image_match * area_hess / pow(area, 2.0);
}

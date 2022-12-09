#include "../../include/Energy/CellFunctionPositionTarget.h"
#include <iostream>

void CellFunctionPositionTarget::addValue(const VectorXT &site, const VectorXT &nodes, double &value) const {
    value += pow(site(0) - target_position(0), 2.0) + pow(site(1) - target_position(1), 2.0);
}

void CellFunctionPositionTarget::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                             VectorXT &gradient_x) const {
    gradient_c(0) += 2 * (site(0) - target_position(0));
    gradient_c(1) += 2 * (site(1) - target_position(1));
}

void CellFunctionPositionTarget::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const {
    hessian(0, 0) += 2;
    hessian(1, 1) += 2;
}

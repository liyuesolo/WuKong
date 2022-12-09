#include "../../include/Tessellation/CellFunctionWeightedSum.h"
#include <iostream>

void CellFunctionWeightedSum::addValue(const VectorXT &site, const VectorXT &nodes, double &value) const {
    for (size_t i = 0; i < functions.size(); i++) {
        double func_value = 0;
        functions[i]->addValue(site, nodes, func_value);
        value += func_value * weights[i];
    }
}

void CellFunctionWeightedSum::addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c,
                                          VectorXT &gradient_x) const {
    for (size_t i = 0; i < functions.size(); i++) {
        VectorXT func_gradient_c = VectorXT::Zero(gradient_c.rows());
        VectorXT func_gradient_x = VectorXT::Zero(gradient_x.rows());
        functions[i]->addGradient(site, nodes, func_gradient_c, func_gradient_x);
        gradient_c += func_gradient_c * weights[i];
        gradient_x += func_gradient_x * weights[i];
    }
}

void CellFunctionWeightedSum::addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const {
    for (size_t i = 0; i < functions.size(); i++) {
        MatrixXT func_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
        functions[i]->addHessian(site, nodes, func_hessian);
        hessian += func_hessian * weights[i];
    }
}

#include "../../include/Tessellation/CellFunctionWeightedSum.h"
#include <iostream>

void CellFunctionWeightedSum::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                       const VectorXi &btype, double &value,
                                       const CellInfo *cellInfo) const {
    for (size_t i = 0; i < functions.size(); i++) {
        if (fabs(weights[i]) < 1e-10) continue;
        double func_value = 0;
        functions[i]->addValue(site, nodes, next, btype, func_value, cellInfo);
        value += func_value * weights[i];
    }
}

void CellFunctionWeightedSum::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                          const VectorXi &btype,
                                          VectorXT &gradient_c,
                                          VectorXT &gradient_x, const CellInfo *cellInfo) const {
    for (size_t i = 0; i < functions.size(); i++) {
        if (fabs(weights[i]) < 1e-10) continue;
        VectorXT func_gradient_c = VectorXT::Zero(gradient_c.rows());
        VectorXT func_gradient_x = VectorXT::Zero(gradient_x.rows());
        functions[i]->addGradient(site, nodes, next, btype, func_gradient_c, func_gradient_x, cellInfo);
        gradient_c += func_gradient_c * weights[i];
        gradient_x += func_gradient_x * weights[i];
    }
}

void CellFunctionWeightedSum::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                         const VectorXi &btype,
                                         MatrixXT &hessian,
                                         const CellInfo *cellInfo) const {
    for (size_t i = 0; i < functions.size(); i++) {
        if (fabs(weights[i]) < 1e-10) continue;
        MatrixXT func_hessian = MatrixXT::Zero(hessian.rows(), hessian.cols());
        functions[i]->addHessian(site, nodes, next, btype, func_hessian, cellInfo);
        hessian += func_hessian * weights[i];
    }
}

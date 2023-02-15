#include "../../include/Energy/CellFunctionPositionTarget.h"
#include <iostream>

void
CellFunctionPositionTarget::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                     const VectorXi &btype, double &value,
                                     const CellInfo *cellInfo) const {
    if (cellInfo->agent) {
        TV target_position = cellInfo->target_position;
        value += pow(site(0) - target_position(0), 2.0) + pow(site(1) - target_position(1), 2.0);
    }
}

void CellFunctionPositionTarget::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                             const VectorXi &btype,
                                             VectorXT &gradient_c,
                                             VectorXT &gradient_x, const CellInfo *cellInfo) const {
    if (cellInfo->agent) {
        TV target_position = cellInfo->target_position;
        gradient_c(0) += 2 * (site(0) - target_position(0));
        gradient_c(1) += 2 * (site(1) - target_position(1));
    }
}

void CellFunctionPositionTarget::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                            const VectorXi &btype,
                                            MatrixXT &hessian,
                                            const CellInfo *cellInfo) const {
    if (cellInfo->agent) {
        hessian(0, 0) += 2;
        hessian(1, 1) += 2;
    }
}

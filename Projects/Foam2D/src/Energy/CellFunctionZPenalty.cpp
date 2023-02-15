#include "../../include/Energy/CellFunctionZPenalty.h"
#include <iostream>

void
CellFunctionZPenalty::addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                               double &value,
                               const CellInfo *cellInfo) const {
    int dims = site.rows();
    if (dims > 2) {
        value += site.segment(2, dims - 2).squaredNorm();
    }
}

void CellFunctionZPenalty::addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                       const VectorXi &btype,
                                       VectorXT &gradient_c,
                                       VectorXT &gradient_x, const CellInfo *cellInfo) const {
    int dims = site.rows();
    for (int i = 2; i < dims; i++) {
        gradient_c(i) += 2 * site(i);
    }
}

void
CellFunctionZPenalty::addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next,
                                 const VectorXi &btype, MatrixXT &hessian,
                                 const CellInfo *cellInfo) const {
    int dims = site.rows();
    for (int i = 2; i < dims; i++) {
        hessian(i, i) += 2;
    }
}

#include "Projects/Foam3D/include/Energy/CellFunctionWPenalty.h"
#include <iostream>

void CellFunctionWPenalty::getValue(Tessellation *tessellation, CellValue &value) const {
    double w = tessellation->c(value.cell.cellIndex * 4 + 3);
    value.value = epsilon * pow(w, 2);
}

void CellFunctionWPenalty::getGradient(Tessellation *tessellation, CellValue &value) const {
    double w = tessellation->c(value.cell.cellIndex * 4 + 3);
    value.gradient.setZero();
    value.gradient.tail<1>()(0) = 2 * epsilon * w;
}

void CellFunctionWPenalty::getHessian(Tessellation *tessellation, CellValue &value) const {
    double w = tessellation->c(value.cell.cellIndex * 4 + 3);
    value.hessian.setZero();
    value.hessian.bottomRightCorner<1, 1>()(0, 0) = 2 * epsilon;
}

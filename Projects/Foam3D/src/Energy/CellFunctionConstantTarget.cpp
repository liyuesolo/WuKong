#include "Projects/Foam3D/include/Energy/CellFunctionConstantTarget.h"

void CellFunctionConstantTarget::getValue(Tessellation *tessellation, CellValue &value) const {
    CellValue internalValue = value;
    internalFunction->getValue(tessellation, internalValue);
    value.value = (internalValue.value - target) * (internalValue.value - target);
}

void CellFunctionConstantTarget::getGradient(Tessellation *tessellation, CellValue &value) const {
    CellValue internalValue = value;
    internalFunction->getValue(tessellation, internalValue);
    internalFunction->getGradient(tessellation, internalValue);
    value.gradient = 2 * (internalValue.value - target) * internalValue.gradient;
}

void CellFunctionConstantTarget::getHessian(Tessellation *tessellation, CellValue &value) const {
    CellValue internalValue = value;
    internalFunction->getValue(tessellation, internalValue);
    internalFunction->getGradient(tessellation, internalValue);
    internalFunction->getHessian(tessellation, internalValue);
    value.hessian = 2 * (internalValue.value - target) * internalValue.hessian +
                    2 * internalValue.gradient * internalValue.gradient.transpose();
}

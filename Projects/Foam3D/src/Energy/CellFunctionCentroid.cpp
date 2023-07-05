#include "Projects/Foam3D/include/Energy/CellFunctionCentroid.h"
#include <iostream>

void CellFunctionCentroid::getValue(Tessellation *tessellation, CellValue &value) const {
    CellValue wm = value;
    wmFunc->getValue(tessellation, wm);
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);

    value.value = wm.value / vol.value;
}

void CellFunctionCentroid::getGradient(Tessellation *tessellation, CellValue &value) const {
    CellValue wm = value;
    wmFunc->getValue(tessellation, wm);
    wmFunc->getGradient(tessellation, wm);
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);

    value.gradient = wm.gradient / vol.value - wm.value * vol.gradient / pow(vol.value, 2.0);
}

void CellFunctionCentroid::getHessian(Tessellation *tessellation, CellValue &value) const {
    CellValue wm = value;
    wmFunc->getValue(tessellation, wm);
    wmFunc->getGradient(tessellation, wm);
    wmFunc->getHessian(tessellation, wm);
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);
    volFunc->getHessian(tessellation, vol);

    value.hessian = wm.hessian / vol.value
                    - wm.gradient * vol.gradient.transpose() / pow(vol.value, 2.0)
                    - vol.gradient * wm.gradient.transpose() / pow(vol.value, 2.0)
                    + 2 * wm.value * vol.gradient * vol.gradient.transpose() / pow(vol.value, 3.0)
                    - wm.value * vol.hessian / pow(vol.value, 2.0);
}

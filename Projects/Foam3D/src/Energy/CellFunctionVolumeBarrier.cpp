#include "Projects/Foam3D/include/Energy/CellFunctionVolumeBarrier.h"
#include <iostream>

void CellFunctionVolumeBarrier::getValue(Tessellation *tessellation, CellValue &value) const {
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    value.value = epsilon * pow(vol.value, exponent);
}

void CellFunctionVolumeBarrier::getGradient(Tessellation *tessellation, CellValue &value) const {
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);
    value.gradient = epsilon * (exponent * pow(vol.value, exponent - 1.0) * vol.gradient);
}

void CellFunctionVolumeBarrier::getHessian(Tessellation *tessellation, CellValue &value) const {
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);
    volFunc->getHessian(tessellation, vol);
    value.hessian = epsilon * (exponent * (exponent - 1.0) * pow(vol.value, exponent - 2.0) * vol.gradient *
                               vol.gradient.transpose()
                               + exponent * pow(vol.value, exponent - 1.0) * vol.hessian);
}

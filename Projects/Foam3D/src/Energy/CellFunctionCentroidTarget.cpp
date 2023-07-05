#include "Projects/Foam3D/include/Energy/CellFunctionCentroidTarget.h"
#include <iostream>

void CellFunctionCentroidTarget::getValue(Tessellation *tessellation, CellValue &value) const {
    TV3 site = tessellation->c.segment<3>(value.cell.cellIndex * 4);
    CellValue wmVals[3] = {value, value, value};
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);

    value.value = 0;
    for (int i = 0; i < 3; i++) {
        CellValue &wm = wmVals[i];
        wmFuncs[i]->getValue(tessellation, wm);
        value.value += pow(wm.value / vol.value - site(i), 2.0);
    }
}

void CellFunctionCentroidTarget::getGradient(Tessellation *tessellation, CellValue &value) const {
    TV3 site = tessellation->c.segment<3>(value.cell.cellIndex * 4);
    CellValue wmVals[3] = {value, value, value};
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);

    value.gradient.setZero();
    for (int i = 0; i < 3; i++) {
        CellValue &wm = wmVals[i];
        wmFuncs[i]->getValue(tessellation, wm);
        wmFuncs[i]->getGradient(tessellation, wm);
        VectorXT siteGrad = VectorXT::Zero(value.gradient.rows());
        siteGrad(siteGrad.rows() - 4 + i) = 1;
        value.gradient += 2 * (wm.value / vol.value - site(i)) *
                          (wm.gradient / vol.value - wm.value * vol.gradient / pow(vol.value, 2.0) - siteGrad);
    }
}

void CellFunctionCentroidTarget::getHessian(Tessellation *tessellation, CellValue &value) const {
    TV3 site = tessellation->c.segment<3>(value.cell.cellIndex * 4);
    CellValue wmVals[3] = {value, value, value};
    CellValue vol = value;
    volFunc->getValue(tessellation, vol);
    volFunc->getGradient(tessellation, vol);
    volFunc->getHessian(tessellation, vol);

    value.hessian.setZero();
    for (int i = 0; i < 3; i++) {
        CellValue &wm = wmVals[i];
        wmFuncs[i]->getValue(tessellation, wm);
        wmFuncs[i]->getGradient(tessellation, wm);
        wmFuncs[i]->getHessian(tessellation, wm);
        VectorXT siteGrad = VectorXT::Zero(value.gradient.rows());
        siteGrad(siteGrad.rows() - 4 + i) = 1;

        auto temp = (wm.gradient / vol.value - wm.value * vol.gradient / pow(vol.value, 2.0) - siteGrad);
        value.hessian += 2 * temp * temp.transpose()
                         + 2 * (wm.value / vol.value - site(i)) *
                           (wm.hessian / vol.value
                            - wm.gradient * vol.gradient.transpose() / pow(vol.value, 2.0)
                            - vol.gradient * wm.gradient.transpose() / pow(vol.value, 2.0)
                            + 2 * wm.value * vol.gradient * vol.gradient.transpose() / pow(vol.value, 3.0)
                            - wm.value * vol.hessian / pow(vol.value, 2.0));
    }
}

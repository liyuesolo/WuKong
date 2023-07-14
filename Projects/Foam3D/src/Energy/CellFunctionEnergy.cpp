#include "Projects/Foam3D/include/Energy/CellFunctionEnergy.h"
#include "../../include/Energy/CellFunctionCentroidTarget.h"
#include "../../include/Energy/CellFunctionConstantTarget.h"
#include "../../include/Energy/CellFunctionVolumeBarrier.h"
#include "../../include/Energy/CellFunctionWPenalty.h"
#include "../../include/Energy/CellFunctionPerTriangle.h"
#include "../../include/Energy/CellFunctionSecondMoment.h"
#include "../../include/Energy/CellFunctionAdhesion.h"
#include "../../include/Energy/PerTriangleVolume.h"
#include "../../include/Energy/PerTriangleSurfaceArea.h"
#include "../../include/Energy/PerTriangleWeightedMeanX.h"
#include "../../include/Energy/PerTriangleWeightedMeanY.h"
#include "../../include/Energy/PerTriangleWeightedMeanZ.h"
#include <iostream>

#define CONDY (false)
#define NTERMS 7

CellFunctionEnergy::CellFunctionEnergy() {
    CellFunction *vol = new CellFunctionPerTriangle(new PerTriangleVolume());
    CellFunction *wmx = new CellFunctionPerTriangle(new PerTriangleWeightedMeanX());
    CellFunction *wmy = new CellFunctionPerTriangle(new PerTriangleWeightedMeanY());
    CellFunction *wmz = new CellFunctionPerTriangle(new PerTriangleWeightedMeanZ());
    CellFunction *cx = new CellFunctionCentroid(wmx, vol);
    CellFunction *cy = new CellFunctionCentroid(wmy, vol);
    CellFunction *cz = new CellFunctionCentroid(wmz, vol);
    volumeTargetFunction = new CellFunctionConstantTarget(vol, 0.35);
    surfaceAreaTargetFunction = new CellFunctionConstantTarget(
            new CellFunctionPerTriangle(new PerTriangleSurfaceArea()), 0);
    siteCentroidFunction = new CellFunctionCentroidTarget(wmx, wmy, wmz, vol);
    volumeBarrierFunction = new CellFunctionVolumeBarrier(vol);
    wPenaltyFunction = new CellFunctionWPenalty();
    secondMomentFunction = new CellFunctionSecondMoment(cx, cy, cz);
    adhesionFunction = new CellFunctionConstantTarget(
            new CellFunctionAdhesion(), 0);
}

void CellFunctionEnergy::getValue(Tessellation *tessellation, CellValue &value) const {
    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    CellFunction *funcs[NTERMS] = {volumeTargetFunction, surfaceAreaTargetFunction, siteCentroidFunction,
                                   volumeBarrierFunction, wPenaltyFunction, secondMomentFunction, adhesionFunction};
    double weights[NTERMS] = {volumeTargetWeight, surfaceAreaTargetWeight, siteCentroidWeight, volumeBarrierWeight,
                              wPenaltyWeight, secondMomentWeight, adhesionWeight};

    value.value = 0;
    for (int i = 0; i < NTERMS; i++) {
        if (CONDY) continue;
        funcs[i]->getValue(tessellation, vals[i]);
//        std::cout << "val " << i << " " << vals[i].value << std::endl;
//        if (std::isnan(vals[i].value) || std::isinf(vals[i].value)) continue;
//        std::cout << weights[i] * vals[i].value << ", ";
        value.value += weights[i] * vals[i].value;
    }
//    std::cout << std::endl;
}

void CellFunctionEnergy::getGradient(Tessellation *tessellation, CellValue &value) const {
    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    CellFunction *funcs[NTERMS] = {volumeTargetFunction, surfaceAreaTargetFunction, siteCentroidFunction,
                                   volumeBarrierFunction, wPenaltyFunction, secondMomentFunction, adhesionFunction};
    double weights[NTERMS] = {volumeTargetWeight, surfaceAreaTargetWeight, siteCentroidWeight, volumeBarrierWeight,
                              wPenaltyWeight, secondMomentWeight, adhesionWeight};

    value.gradient.setZero();
    for (int i = 0; i < NTERMS; i++) {
        if (CONDY) continue;
        funcs[i]->getGradient(tessellation, vals[i]);
//        std::cout << "grad " << i << " " << vals[i].gradient.norm() << std::endl;
//        if (std::isnan(vals[i].gradient.norm()) || std::isinf(vals[i].gradient.norm())) continue;
        value.gradient += weights[i] * vals[i].gradient;
    }
}

void CellFunctionEnergy::getHessian(Tessellation *tessellation, CellValue &value) const {
    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    CellFunction *funcs[NTERMS] = {volumeTargetFunction, surfaceAreaTargetFunction, siteCentroidFunction,
                                   volumeBarrierFunction, wPenaltyFunction, secondMomentFunction, adhesionFunction};
    double weights[NTERMS] = {volumeTargetWeight, surfaceAreaTargetWeight, siteCentroidWeight, volumeBarrierWeight,
                              wPenaltyWeight, secondMomentWeight, adhesionWeight};

    value.hessian.setZero();
    for (int i = 0; i < NTERMS; i++) {
        if (CONDY) continue;
        funcs[i]->getHessian(tessellation, vals[i]);
        value.hessian += weights[i] * vals[i].hessian;
    }
}

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

#include "../../include/Globals.h"

#define CONDY (false)
#define NTERMS 7

CellFunctionEnergy::CellFunctionEnergy() :
        cellFunctionVolume(&perTriangleVolume),
        cellFunctionSurfaceArea(&perTriangleSurfaceArea),
        cellFunctionWMX(&perTriangleWMX),
        cellFunctionWMY(&perTriangleWMY),
        cellFunctionWMZ(&perTriangleWMZ),
        cellFunctionVolumeTarget(&cellFunctionVolume, 0),
        cellFunctionSurfaceAreaTarget(&cellFunctionSurfaceArea, 0),
        cellFunctionCX(&cellFunctionWMX, &cellFunctionVolume),
        cellFunctionCY(&cellFunctionWMY, &cellFunctionVolume),
        cellFunctionCZ(&cellFunctionWMZ, &cellFunctionVolume),
        cellFunctionCentroidTarget(&cellFunctionWMX, &cellFunctionWMY, &cellFunctionWMZ, &cellFunctionVolume),
        cellFunctionVolumeBarrier(&cellFunctionVolume),
        cellFunctionSecondMoment(&cellFunctionCX, &cellFunctionCY, &cellFunctionCZ),
        cellFunctionAdhesionTarget(&cellFunctionAdhesion, 0) {}

void CellFunctionEnergy::getParameters() const {
    cellFunctionVolumeTarget.target = 4.0 / 3.0 * M_PI * pow(cellRadiusTarget, 3.0);
}

void CellFunctionEnergy::getValue(Tessellation *tessellation, CellValue &value) const {
    getParameters();

    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    const CellFunction *funcs[NTERMS] = {&cellFunctionVolumeTarget, &cellFunctionSurfaceAreaTarget,
                                         &cellFunctionCentroidTarget,
                                         &cellFunctionVolumeBarrier, &cellFunctionWPenalty, &cellFunctionSecondMoment,
                                         &cellFunctionAdhesionTarget};
    double weights[NTERMS] = {volumeTargetWeight, surfaceAreaTargetWeight, siteCentroidWeight, volumeBarrierWeight,
                              wPenaltyWeight, secondMomentWeight, adhesionWeight};

    value.value = 0;
    for (int i = 0; i < NTERMS; i++) {
        if (CONDY) continue;
        funcs[i]->getValue(tessellation, vals[i]);
//        if (tessellation->cellInfos[value.cell.cellIndex].adhesion > 0)
//            std::cout << "val " << i << " " << vals[i].value << std::endl;
//        if (std::isnan(vals[i].value) || std::isinf(vals[i].value)) continue;
//        std::cout << weights[i] * vals[i].value << ", ";
        value.value += weights[i] * vals[i].value;
    }
//    std::cout << std::endl;
}

void CellFunctionEnergy::getGradient(Tessellation *tessellation, CellValue &value) const {
    getParameters();

    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    const CellFunction *funcs[NTERMS] = {&cellFunctionVolumeTarget, &cellFunctionSurfaceAreaTarget,
                                         &cellFunctionCentroidTarget,
                                         &cellFunctionVolumeBarrier, &cellFunctionWPenalty, &cellFunctionSecondMoment,
                                         &cellFunctionAdhesionTarget};
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
    getParameters();

    CellValue vals[NTERMS] = {value, value, value, value, value, value, value};
    const CellFunction *funcs[NTERMS] = {&cellFunctionVolumeTarget, &cellFunctionSurfaceAreaTarget,
                                         &cellFunctionCentroidTarget,
                                         &cellFunctionVolumeBarrier, &cellFunctionWPenalty, &cellFunctionSecondMoment,
                                         &cellFunctionAdhesionTarget};
    double weights[NTERMS] = {volumeTargetWeight, surfaceAreaTargetWeight, siteCentroidWeight, volumeBarrierWeight,
                              wPenaltyWeight, secondMomentWeight, adhesionWeight};

    value.hessian.setZero();
    for (int i = 0; i < NTERMS; i++) {
        if (CONDY) continue;
        funcs[i]->getHessian(tessellation, vals[i]);
        value.hessian += weights[i] * vals[i].hessian;
    }
}

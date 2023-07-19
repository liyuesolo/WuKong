#pragma once

#include "CellFunction.h"
#include "../../include/Energy/PerTriangleVolume.h"
#include "../../include/Energy/PerTriangleSurfaceArea.h"
#include "../../include/Energy/PerTriangleWeightedMeanX.h"
#include "../../include/Energy/PerTriangleWeightedMeanY.h"
#include "../../include/Energy/PerTriangleWeightedMeanZ.h"
#include "../../include/Energy/CellFunctionCentroidTarget.h"
#include "../../include/Energy/CellFunctionConstantTarget.h"
#include "../../include/Energy/CellFunctionVolumeBarrier.h"
#include "../../include/Energy/CellFunctionWPenalty.h"
#include "../../include/Energy/CellFunctionPerTriangle.h"
#include "../../include/Energy/CellFunctionSecondMoment.h"
#include "../../include/Energy/CellFunctionAdhesion.h"

class CellFunctionEnergy : public CellFunction {
public:
    PerTriangleVolume perTriangleVolume;
    PerTriangleSurfaceArea perTriangleSurfaceArea;
    PerTriangleWeightedMeanX perTriangleWMX;
    PerTriangleWeightedMeanY perTriangleWMY;
    PerTriangleWeightedMeanZ perTriangleWMZ;
    CellFunctionPerTriangle cellFunctionVolume;
    CellFunctionPerTriangle cellFunctionSurfaceArea;
    CellFunctionPerTriangle cellFunctionWMX;
    CellFunctionPerTriangle cellFunctionWMY;
    CellFunctionPerTriangle cellFunctionWMZ;
    mutable CellFunctionConstantTarget cellFunctionVolumeTarget;
    CellFunctionConstantTarget cellFunctionSurfaceAreaTarget;
    CellFunctionCentroid cellFunctionCX;
    CellFunctionCentroid cellFunctionCY;
    CellFunctionCentroid cellFunctionCZ;
    CellFunctionCentroidTarget cellFunctionCentroidTarget;
    CellFunctionVolumeBarrier cellFunctionVolumeBarrier;
    CellFunctionWPenalty cellFunctionWPenalty;
    CellFunctionSecondMoment cellFunctionSecondMoment;
    CellFunctionAdhesion cellFunctionAdhesion;
    CellFunctionConstantTarget cellFunctionAdhesionTarget;

//    double volumeTargetWeight = 500;//100;
//    double surfaceAreaTargetWeight = 0.0;
//    double siteCentroidWeight = 1;//1;
//    double volumeBarrierWeight = 1;
//    double wPenaltyWeight = 1;
//    double secondMomentWeight = 500;//100;
//    double adhesionWeight = 50;
//
//    double cellRadiusTarget = 0.4;

//    double volumeTargetWeight = 100;
//    double surfaceAreaTargetWeight = 0;
//    double siteCentroidWeight = 0;
//    double volumeBarrierWeight = 0;
//    double wPenaltyWeight = 0;
//    double secondMomentWeight = 0;
//    double adhesionWeight = 0;

private:
    void getParameters() const;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionEnergy();
};

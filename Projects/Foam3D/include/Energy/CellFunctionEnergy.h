#pragma once

#include "CellFunction.h"

class CellFunctionEnergy : public CellFunction {
public:
    CellFunction *volumeTargetFunction;
    CellFunction *surfaceAreaTargetFunction;
    CellFunction *siteCentroidFunction;
    CellFunction *volumeBarrierFunction;
    CellFunction *wPenaltyFunction;
    CellFunction *secondMomentFunction;

    double volumeTargetWeight = 100;//100;
    double surfaceAreaTargetWeight = 0.0;
    double siteCentroidWeight = 1;//1;
    double volumeBarrierWeight = 1;
    double wPenaltyWeight = 1;
    double secondMomentWeight = 100;

//    double volumeTargetWeight = 100;
//    double surfaceAreaTargetWeight = 0;
//    double siteCentroidWeight = 0;
//    double volumeBarrierWeight = 0;
//    double wPenaltyWeight = 0;
//    double secondMomentWeight = 0;

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

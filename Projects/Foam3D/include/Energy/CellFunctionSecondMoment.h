#pragma once

#include "CellFunction.h"
#include "PerTriangleSecondMoment.h"
#include "CellFunctionCentroid.h"

class CellFunctionSecondMoment : public CellFunction {
public:
    PerTriangleSecondMoment perTriangleFunction;
    CellFunction *centroidFuncs[3];

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionSecondMoment(CellFunction *cx_, CellFunction *cy_, CellFunction *cz_) {
        centroidFuncs[0] = cx_;
        centroidFuncs[1] = cy_;
        centroidFuncs[2] = cz_;
    }
};

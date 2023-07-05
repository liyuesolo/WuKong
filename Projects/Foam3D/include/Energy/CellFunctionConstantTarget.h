#pragma once

#include "CellFunctionPerTriangle.h"
#include "PerTriangleVolume.h"

class CellFunctionConstantTarget : public CellFunction {
public:
    CellFunction *internalFunction;
    double target;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionConstantTarget(CellFunction *internal, double target_) {
        internalFunction = internal;
        target = target_;
    }
};

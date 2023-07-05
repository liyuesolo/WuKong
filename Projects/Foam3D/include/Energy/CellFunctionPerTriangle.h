#pragma once

#include "CellFunction.h"
#include "PerTriangleFunction.h"

class CellFunctionPerTriangle : public CellFunction {
public:
    PerTriangleFunction *perTriangleFunction;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionPerTriangle(PerTriangleFunction *internal) {
        perTriangleFunction = internal;
    }
};

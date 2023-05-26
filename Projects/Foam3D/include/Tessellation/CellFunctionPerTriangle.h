#pragma once

#include "CellFunction.h"
#include "PerTriangleFunction.h"

class CellFunctionPerTriangle : public CellFunction {
public:
    PerTriangleFunction *perTriangleFunction;

public:
    virtual void
    addValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    addGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    addHessian(Tessellation *tessellation, CellValue &value) const;
};

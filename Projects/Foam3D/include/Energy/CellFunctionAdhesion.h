#pragma once

#include "CellFunction.h"
#include "PerTriangleSurfaceArea.h"

class CellFunctionAdhesion : public CellFunction {
public:
    PerTriangleSurfaceArea perTriangleFunction;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
};

#pragma once

#include "CellFunction.h"

class CellFunctionWPenalty : public CellFunction {
public:
    double epsilon = 1e-3;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;
};

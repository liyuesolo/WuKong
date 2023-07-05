#pragma once

#include "CellFunction.h"

class CellFunctionCentroid : public CellFunction {
public:
    CellFunction *wmFunc;
    CellFunction *volFunc;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionCentroid(CellFunction *wm_, CellFunction *vol_) {
        wmFunc = wm_;
        volFunc = vol_;
    }
};

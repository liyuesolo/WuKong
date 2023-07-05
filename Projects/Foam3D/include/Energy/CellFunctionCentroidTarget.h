#pragma once

#include "CellFunction.h"

class CellFunctionCentroidTarget : public CellFunction {
public:
    CellFunction *wmFuncs[3];
    CellFunction *volFunc;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionCentroidTarget(CellFunction *wmx_, CellFunction *wmy_, CellFunction *wmz_, CellFunction *vol_) {
        wmFuncs[0] = wmx_;
        wmFuncs[1] = wmy_;
        wmFuncs[2] = wmz_;
        volFunc = vol_;
    }
};

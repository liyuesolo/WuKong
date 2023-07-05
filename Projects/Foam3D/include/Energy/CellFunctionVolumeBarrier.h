#pragma once

#include "CellFunctionPerTriangle.h"
#include "PerTriangleVolume.h"

class CellFunctionVolumeBarrier : public CellFunction {
public:
    CellFunction *volFunc;
    double epsilon = 1e-18;
    double exponent = -3;

public:
    virtual void
    getValue(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getGradient(Tessellation *tessellation, CellValue &value) const;

    virtual void
    getHessian(Tessellation *tessellation, CellValue &value) const;

public:
    CellFunctionVolumeBarrier(CellFunction *vol_) {
        volFunc = vol_;
    }
};

#pragma once

#include "../../include/Tessellation/PerTriangleFunction.h"

class PerTriangleVolume : public PerTriangleFunction {
public:
    virtual void addValue(TriangleValue &value) const;

    virtual void addGradient(TriangleValue &value) const;

    virtual void addHessian(TriangleValue &value) const;
};

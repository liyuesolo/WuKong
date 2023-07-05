#pragma once

#include "PerTriangleFunction.h"

class PerTriangleSurfaceArea : public PerTriangleFunction {
public:
    virtual void getValue(TriangleValue &value) const;

    virtual void getGradient(TriangleValue &value) const;

    virtual void getHessian(TriangleValue &value) const;

    virtual bool flipSignForBackface() const { return false; };
};

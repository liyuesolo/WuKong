#pragma once

#include "PerTriangleFunction.h"

class PerTriangleSecondMoment {
public:
    void getValue(TriangleValue &value, const TV3 &centroid) const;

    void getGradient(TriangleValue &value, const TV3 &centroid) const;

    void getHessian(TriangleValue &value, const TV3 &centroid) const;

    bool flipSignForBackface() const { return true; };
};

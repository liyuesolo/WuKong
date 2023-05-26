#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam2D/include/VecMatDef.h"

struct TriangleValue {
    TV3 v0;
    TV3 v1;
    TV3 v2;

    double value;
    VectorXT gradient;
    MatrixXT hessian;
};

class PerTriangleFunction {
public:
    virtual void
    addValue(TriangleValue &value) const = 0;

    virtual void
    addGradient(TriangleValue &value) const = 0;

    virtual void
    addHessian(TriangleValue &value) const = 0;
};

#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"

struct NeighborhoodValue {
    TV3 c;
    std::vector<TV3> v;
    int ic;
    std::vector<int> iv;

    double value;
    VectorXT gradient;
    MatrixXT hessian;
};

class NeighborhoodFunction {
public:
    virtual void
    getValue(NeighborhoodValue &value) const = 0;

    virtual void
    getGradient(NeighborhoodValue &value) const = 0;

    virtual void
    getHessian(NeighborhoodValue &value) const = 0;
};

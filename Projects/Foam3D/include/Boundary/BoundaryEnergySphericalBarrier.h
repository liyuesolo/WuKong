#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include "Boundary.h"

class BoundaryEnergySphericalBarrier {
    double eps = 1e-7;
    double exponent = -2.0;
    double radius = 2.55;
    TV3 center;
public:
    virtual void
    getValue(Boundary *boundary, double &value) const;

    virtual void
    getGradient(Boundary *boundary, VectorXT &gradient) const;

    virtual void
    getHessian(Boundary *boundary, MatrixXT &hessian) const;

    BoundaryEnergySphericalBarrier(double radius_, TV3 center_) : radius(radius_), center(center_) {}
};

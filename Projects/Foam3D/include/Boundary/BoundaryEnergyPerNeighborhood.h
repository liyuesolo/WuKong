#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include "NeighborhoodFunction.h"
#include "Boundary.h"

class BoundaryEnergyPerNeighborhood {
    NeighborhoodFunction *internalFunction;
public:
    virtual void
    getValue(Boundary *boundary, double &value) const;

    virtual void
    getGradient(Boundary *boundary, VectorXT &gradient) const;

    virtual void
    getHessian(Boundary *boundary, MatrixXT &hessian) const;

    BoundaryEnergyPerNeighborhood(NeighborhoodFunction *internal) : internalFunction(internal) {}
};

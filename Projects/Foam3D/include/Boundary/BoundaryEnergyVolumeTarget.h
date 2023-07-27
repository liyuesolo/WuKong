#pragma once

#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "Projects/Foam3D/include/VecMatDef.h"
#include "Boundary.h"

class BoundaryEnergyVolumeTarget {
    double target;
    std::vector<int> faces;
public:
    virtual void
    getValue(Boundary *boundary, double &value) const;

    virtual void
    getGradient(Boundary *boundary, VectorXT &gradient) const;

    virtual void
    getHessian(Boundary *boundary, MatrixXT &hessian) const;

    virtual void
    getHessianWoodbury(Boundary *boundary, Eigen::SparseMatrix<double> &K, VectorXT &UV) const;

    BoundaryEnergyVolumeTarget(double target_, std::vector<int> faces_) : target(target_), faces(faces_) {}
};

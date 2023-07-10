#pragma once

#include "Boundary.h"

class CubeBoundary : public Boundary {


private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual VectorXT computeEnergyGradient();

    virtual MatrixXT computeEnergyHessian();

public:
    CubeBoundary(double r, const VectorXi &free_);
};

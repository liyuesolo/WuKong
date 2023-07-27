#pragma once

#include "Boundary.h"

class CubeBoundary : public Boundary {


private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual void computeEnergyGradient(VectorXT &gradient);

    virtual void computeEnergyHessian(MatrixXT &hessian);

public:
    CubeBoundary(double r, const VectorXi &free_);
};

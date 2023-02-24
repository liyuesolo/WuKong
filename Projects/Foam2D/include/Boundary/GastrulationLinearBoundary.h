#pragma once

#include "Boundary.h"

class GastrulationLinearBoundary : public Boundary {

private:
    double w = 1e-4;

private:
    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual VectorXT computeEnergyGradient();

    virtual MatrixXT computeEnergyHessian();

public:
    GastrulationLinearBoundary(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

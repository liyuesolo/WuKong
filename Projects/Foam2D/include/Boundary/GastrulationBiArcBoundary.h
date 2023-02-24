#pragma once

#include "Boundary.h"

class GastrulationBiArcBoundary : public Boundary {

private:
    double epsilon = 0 * 1e-8;

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
    GastrulationBiArcBoundary(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

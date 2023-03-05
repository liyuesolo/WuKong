#pragma once

#include "Boundary.h"

class HardwareBoundary0 : public Boundary {

public:
    double channel_width = 0.8;
    double corner_radius = 0.13333333333333333;
    double w_barrier = 1e-7;
    double w_piston = 1e-2;

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
    HardwareBoundary0(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

#pragma once

#include "Boundary.h"

class GastrulationBezierBoundary : public Boundary {

private:
    double yolk_area = M_PI * 0.5 * 0.5;
    double membrane_radius = 0.8;

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    virtual bool checkValid();

public:
    GastrulationBezierBoundary(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

#pragma once

#include "Boundary.h"

class RegularPolygonBoundary : public Boundary {

public:
    int nsides;

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    RegularPolygonBoundary(const VectorXT &p_, const VectorXi &free_, const int n) : Boundary(p_, free_), nsides(n) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

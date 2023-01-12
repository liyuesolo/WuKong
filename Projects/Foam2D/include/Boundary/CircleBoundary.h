#pragma once

#include "Boundary.h"

class CircleBoundary : public Boundary {

public:
    int nsides;

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    CircleBoundary(const VectorXT &p_, const VectorXi &free_, const int n) : Boundary(p_, free_), nsides(n) {}
};

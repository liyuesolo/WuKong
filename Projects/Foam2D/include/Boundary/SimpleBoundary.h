#pragma once

#include "Boundary.h"

class SimpleBoundary : public Boundary {

public:

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    SimpleBoundary(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

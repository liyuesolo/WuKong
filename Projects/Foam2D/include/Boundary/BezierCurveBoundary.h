#pragma once

#include "Boundary.h"

class BezierCurveBoundary : public Boundary {

public:

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    virtual bool checkValid();

public:
    BezierCurveBoundary(const VectorXT &p_, const VectorXi &free_) : Boundary(p_, free_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

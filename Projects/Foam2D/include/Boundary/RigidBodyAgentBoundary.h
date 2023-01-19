#pragma once

#include "Boundary.h"

class RigidBodyAgentBoundary : public Boundary {

public:
    VectorXT agentShape;

private:

    virtual void computeVertices();

    virtual void computeGradient();

    virtual void computeHessian();

public:
    RigidBodyAgentBoundary(const VectorXT &p_, const VectorXi &free_, VectorXT agentShape_) : Boundary(p_, free_),
                                                                                              agentShape(agentShape_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

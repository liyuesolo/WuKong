#pragma once

#include "Boundary.h"

class RigidBodyAgentBoundary : public Boundary {

public:
    VectorXT agentShape;
    double bx = 0.75;
    double by = 0.75;
    double epsilon = 1e-5;

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
    RigidBodyAgentBoundary(const VectorXT &p_, const VectorXi &free_, VectorXT agentShape_) : Boundary(p_, free_),
                                                                                              agentShape(agentShape_) {
        computeVertices();
        computeGradient();
        computeHessian();
    }
};

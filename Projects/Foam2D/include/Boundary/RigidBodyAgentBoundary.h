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
    RigidBodyAgentBoundary(const VectorXT &p_, const VectorXi &free_, VectorXT agentShape_, VectorXT r_,
                           VectorXi r_map_) : Boundary(p_, free_),
                                              agentShape(agentShape_) {
        radii = r_;
        r_map = r_map_;

        computeVertices();
        computeGradient();
        computeHessian();
    }
};

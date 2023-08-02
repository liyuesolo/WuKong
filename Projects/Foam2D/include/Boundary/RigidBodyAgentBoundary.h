#pragma once

#include "Boundary.h"

class RigidBodyAgentBoundary : public Boundary {

public:
    VectorXT agentShape;
    double bx = 0.8;
    double by = 0.8;
    double epsilon = 1e-5;
    double tmul = 10;

    VectorXi q_map;

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
    RigidBodyAgentBoundary(const VectorXT &p_, const VectorXi &free_, VectorXT agentShape_, VectorXT q_,
                           VectorXi q_map_) : Boundary(p_, free_),
                                              agentShape(agentShape_) {
        q = q_;
        q_map = q_map_;

        computeVertices();
        computeGradient();
        computeHessian();
    }
};

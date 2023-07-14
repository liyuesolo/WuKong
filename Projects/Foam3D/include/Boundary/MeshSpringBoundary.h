#pragma once

#include "Boundary.h"

class MeshSpringBoundary : public Boundary {

private:
    double kEdge = 0.1;
    double kVol = 1;
    double volTarget = -48;

private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual VectorXT computeEnergyGradient();

    virtual MatrixXT computeEnergyHessian();

public:
    MeshSpringBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_);
};

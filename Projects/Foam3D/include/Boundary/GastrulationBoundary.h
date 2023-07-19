#pragma once

#include "Boundary.h"

class GastrulationBoundary : public Boundary {

private:
//    double kNeighborhood = 2;
//    double kVol = 2;
//    double volTarget = -40.666;

private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual VectorXT computeEnergyGradient();

    virtual MatrixXT computeEnergyHessian();

public:
    GastrulationBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_);
};

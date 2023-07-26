#pragma once

#include "SubdivisionMeshBoundary.h"

class GastrulationBoundary : public SubdivisionMeshBoundary {

private:
//    double kNeighborhood = 2;
//    double kVol = 2;
//    double volTarget = -40.666;

    bool dvdp_is_identity = true;

public:
    virtual bool checkValid();

    virtual double computeEnergy();

    virtual VectorXT computeEnergyGradient();

    virtual MatrixXT computeEnergyHessian();

public:
    GastrulationBoundary(MatrixXT &v_, MatrixXi &f_, const VectorXi &free_);
};

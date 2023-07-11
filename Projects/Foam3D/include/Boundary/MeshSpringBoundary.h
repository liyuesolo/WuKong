#pragma once

#include "Boundary.h"

struct BoundaryEdgeSpring {
    int v0;
    int v1;
    double len;

    BoundaryEdgeSpring(int v0_, int v1_, double len_) : v0(v0_), v1(v1_), len(len_) {}
};

class MeshSpringBoundary : public Boundary {

private:
    std::vector<BoundaryEdgeSpring> springs;
    double k = 100;

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

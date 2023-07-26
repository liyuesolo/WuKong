#pragma once

#include "MeshBoundary.h"

class SubdivisionMeshBoundary : public MeshBoundary {
public:
    int nSub;
    Eigen::SparseMatrix<double> S;

private:
    virtual void computeVertices();

public:
    SubdivisionMeshBoundary(MatrixXT &v_, MatrixXi &f_, const VectorXi &free_, int nSubdivision = 1);
};

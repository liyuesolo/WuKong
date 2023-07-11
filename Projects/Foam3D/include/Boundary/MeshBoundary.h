#pragma once

#include "Boundary.h"

class MeshBoundary : public Boundary {

private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

public:
    MeshBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_);
};

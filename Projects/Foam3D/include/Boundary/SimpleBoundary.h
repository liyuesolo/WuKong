#pragma once

#include "Boundary.h"

class SimpleBoundary : public Boundary {

private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

public:
    SimpleBoundary(MatrixXT &v_, const MatrixXi &f_, const VectorXi &free_);
};

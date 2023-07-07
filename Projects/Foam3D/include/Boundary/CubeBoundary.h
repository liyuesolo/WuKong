#pragma once

#include "Boundary.h"

class CubeBoundary : public Boundary {

private:
    virtual void computeVertices();

public:
    virtual bool checkValid();

public:
    CubeBoundary(double r, const VectorXi &free_);
};

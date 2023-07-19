#pragma once

#include "NeighborhoodFunction.h"

class NeighborhoodAveragePos : public NeighborhoodFunction {
    int coord;
public:
    virtual void getValue(NeighborhoodValue &value) const;

    virtual void getGradient(NeighborhoodValue &value) const;

    virtual void getHessian(NeighborhoodValue &value) const;

    NeighborhoodAveragePos(int coord_) : coord(coord_) {}
};

#pragma once

#include "NeighborhoodAveragePos.h"

class NeighborhoodLaplacian : public NeighborhoodFunction {
    NeighborhoodAveragePos averagePosFunc[3];
public:
    virtual void getValue(NeighborhoodValue &value) const;

    virtual void getGradient(NeighborhoodValue &value) const;

    virtual void getHessian(NeighborhoodValue &value) const;

    NeighborhoodLaplacian() : averagePosFunc{NeighborhoodAveragePos(0), NeighborhoodAveragePos(1),
                                             NeighborhoodAveragePos(2)} {}
};

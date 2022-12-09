#pragma once

#include "../../include/Tessellation/CellFunction.h"

class CellFunctionPerimeter : public CellFunction {
    double epsilon = 1e-14;

public:
    virtual void addValue(const VectorXT &site, const VectorXT &nodes, double &value) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const;
};

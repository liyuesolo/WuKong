#pragma once

#include "../../include/Tessellation/CellFunction.h"

class CellFunctionCentroidY : public CellFunction {
public:
    virtual void addValue(const VectorXT &site, const VectorXT &nodes, double &value) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const;
};

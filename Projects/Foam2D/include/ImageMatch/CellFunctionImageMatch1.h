#pragma once

#include "../../include/Tessellation/CellFunction.h"

class CellFunctionImageMatch1 : public CellFunction {
    double epsilon = 1e-14;
    double beta = 100;

public:
    virtual void
    addValue(const VectorXT &site, const VectorXT &nodes, double &value, const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian, const CellInfo *cellInfo) const;
};

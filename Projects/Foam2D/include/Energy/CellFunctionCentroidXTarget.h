#pragma once

#include "../../include/Energy/CellFunctionArea.h"
#include "../../include/Energy/CellFunctionCentroidX.h"

class CellFunctionCentroidXTarget : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionCentroidX centroid_function;

public:
    virtual void addValue(const VectorXT &site, const VectorXT &nodes, double &value) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian) const;
};

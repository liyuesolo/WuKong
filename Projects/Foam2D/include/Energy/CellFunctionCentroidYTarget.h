#pragma once

#include "../../include/Energy/CellFunctionArea.h"
#include "../../include/Energy/CellFunctionCentroidY.h"

class CellFunctionCentroidYTarget : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionCentroidY centroid_function;

public:
    virtual void addValue(const VectorXT &site, const VectorXT &nodes, double &value, const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian, const CellInfo *cellInfo) const;
};

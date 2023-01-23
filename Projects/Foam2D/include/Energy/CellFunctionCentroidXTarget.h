#pragma once

#include "../../include/Energy/CellFunctionArea.h"
#include "../../include/Energy/CellFunctionCentroidX.h"

class CellFunctionCentroidXTarget : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionCentroidX centroid_function;

public:
    virtual void addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
                          const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_c,
                VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
               const CellInfo *cellInfo) const;
};

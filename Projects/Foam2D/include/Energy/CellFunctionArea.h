#pragma once

#include "../../include/Tessellation/CellFunction.h"

class CellFunctionArea : public CellFunction {
public:
    virtual void
    addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, double &value,
             const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_c,
                VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, MatrixXT &hessian,
               const CellInfo *cellInfo) const;
};

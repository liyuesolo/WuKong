#pragma once

#include "../../include/Energy/CellFunctionCentroidX.h"
#include "../../include/Energy/CellFunctionCentroidY.h"

class CellFunctionSecondMoment : public CellFunction {
    CellFunctionCentroidX xc_function;
    CellFunctionCentroidY yc_function;

public:
    virtual void
    addValue(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype, double &value,
             const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                VectorXT &gradient_c,
                VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
               MatrixXT &hessian,
               const CellInfo *cellInfo) const;
};

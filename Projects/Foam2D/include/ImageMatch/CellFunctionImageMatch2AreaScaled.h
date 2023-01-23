#pragma once

#include "../../include/Tessellation/CellFunction.h"
#include "../../include/ImageMatch/CellFunctionImageMatch2.h"
#include "../../include/Energy/CellFunctionArea.h"

class CellFunctionImageMatch2AreaScaled : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionImageMatch2 image_match_function;

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

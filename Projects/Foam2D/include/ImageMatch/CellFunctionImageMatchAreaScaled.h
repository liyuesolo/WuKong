#pragma once

#include "../../include/Tessellation/CellFunction.h"
#include "../../include/ImageMatch/CellFunctionImageMatch.h"
#include "../../include/Energy/CellFunctionArea.h"

class CellFunctionImageMatchAreaScaled : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionImageMatch image_match_function;

public:
    virtual void
    addValue(const VectorXT &site, const VectorXT &nodes, double &value, const CellInfo *cellInfo) const;

    virtual void
    addGradient(const VectorXT &site, const VectorXT &nodes, VectorXT &gradient_c, VectorXT &gradient_x,
                const CellInfo *cellInfo) const;

    virtual void
    addHessian(const VectorXT &site, const VectorXT &nodes, MatrixXT &hessian, const CellInfo *cellInfo) const;
};

#pragma once

#include "../../include/Tessellation/CellFunction.h"
#include "../../include/ImageMatch/CellFunctionImageMatch1.h"
#include "../../include/Energy/CellFunctionArea.h"

class CellFunctionImageMatch1AreaScaled : public CellFunction {
public:
    CellFunctionArea area_function;
    CellFunctionImageMatch1 image_match_function;

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
    addHessian(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype, MatrixXT &hessian,
               const CellInfo *cellInfo) const;
};

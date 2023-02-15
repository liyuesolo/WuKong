#pragma once

#include "../../include/Energy/CellFunctionArea.h"

class CellFunctionAreaTarget : public CellFunction {
public:
    CellFunctionArea area_function;

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

    void addGradient_tau(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                         double &gradient_tau,
                         const CellInfo *cellInfo) const;

    void addHessian_tau(const VectorXT &site, const VectorXT &nodes, const VectorXi &next, const VectorXi &btype,
                        double &hessian_tau_tau,
                        VectorXT &hessian_tau_x,
                        const CellInfo *cellInfo) const;
};

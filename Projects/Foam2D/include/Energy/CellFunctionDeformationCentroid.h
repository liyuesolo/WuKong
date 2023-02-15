#pragma once

#include "../../include/Energy/CellFunctionCentroidX.h"
#include "../../include/Energy/CellFunctionCentroidY.h"

class CellFunctionDeformationCentroid : public CellFunction {
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

private:
    void getArcParams(int order, const VectorXT &x, VectorXT &p, MatrixXT &dpdx, std::vector<MatrixXT> &d2pdx2) const;

    void
    addValueArc(int i, const VectorXT &nodes, const VectorXi &next, double &value, const CellInfo *cellInfo, double xc,
                double yc) const;

    void
    addGradientArc(int i, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_x,
                   VectorXT &gradient_centroid,
                   const CellInfo *cellInfo, double xc,
                   double yc) const;

    void
    addHessianArc(int i, const VectorXT &nodes, const VectorXi &next, VectorXT &gradient_centroid,
                  Eigen::Ref<MatrixXT> &hess_xx,
                  MatrixXT &hess_Cx, MatrixXT &hess_CC,
                  const CellInfo *cellInfo, double xc,
                  double yc) const;
};

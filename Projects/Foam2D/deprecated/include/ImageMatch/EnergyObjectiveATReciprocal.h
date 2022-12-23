#pragma once

#include "Projects/Foam2D/src/optLib/ObjectiveFunction.h"
#include "Projects/Foam2D/include/Foam2DInfo.h"

class EnergyObjectiveATReciprocal : public ObjectiveFunction {

public:
    Foam2DInfo *info;

public:

    void preProcess(const VectorXd &x, std::vector<CellInfo> &cellInfos) const;

    virtual double evaluate(const VectorXd &x) const;

    virtual void addGradientTo(const VectorXd &x, VectorXd &grad) const;

    VectorXd get_dOdx(const VectorXd &x) const;

    virtual void getHessian(const VectorXd &x, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odx2(const VectorXd &x) const;
};

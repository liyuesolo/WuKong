#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"

class EnergyObjectiveAT : public ObjectiveFunction {

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

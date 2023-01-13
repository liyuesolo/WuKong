#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"

class EnergyObjective : public ObjectiveFunction {

public:
    Foam2DInfo *info;

public:

    void check_gradients(const VectorXd &y) const;

    void preProcess(const VectorXd &y, std::vector<CellInfo> &cellInfos) const;

    virtual double evaluate(const VectorXd &y) const;

    virtual void addGradientTo(const VectorXd &y, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &y) const;

    virtual void getHessian(const VectorXd &y, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &y) const;
};

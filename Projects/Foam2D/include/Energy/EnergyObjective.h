#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"

class EnergyObjective : public ObjectiveFunction {

public:
    Foam2DInfo *info;

public:

    void check_gradients(const VectorXd &c_free) const;

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &c_free) const;
};

#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"

class EnergyObjective : public ObjectiveFunction {

public:
    Foam2DInfo *info;

public:
    void
    getInputs(const VectorXT &vertices, const int cellIndex, std::vector<int> cell, VectorXT &p_in,
              VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
              VectorXi &map) const;

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &c_free) const;
};

#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"

class EnergyObjectiveAT : public ObjectiveFunction {

public:
    Foam2DInfo *info;

public:
    void
    getInputs(const VectorXT &c, const VectorXT &area_targets, const int cellIndex, std::vector<int> cell,
              VectorXT &p_in, VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
              VectorXi &map) const;

    virtual double evaluate(const VectorXd &x) const;

    virtual void addGradientTo(const VectorXd &x, VectorXd &grad) const;

    VectorXd get_dOdx(const VectorXd &x) const;

    virtual void getHessian(const VectorXd &x, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odx2(const VectorXd &x) const;
};

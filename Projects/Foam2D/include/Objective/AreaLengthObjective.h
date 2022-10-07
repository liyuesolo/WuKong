#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"

class AreaLengthObjective : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    VectorXd c_fixed;

    VectorXd area_targets = 0.05 * VectorXd::Ones(1);
    double area_weight = 0.1;
    double length_weight = 0.01;
    double centroid_weight = 0.05;

    int n_free;
    int n_fixed;

public:
    void
    getInputs(const VectorXT &vertices, const int cellIndex, std::vector<int> cell, VectorXT &c_cell, VectorXT &p_cell,
              VectorXi &i_cell) const;

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &c_free) const;

    double getAreaTarget(int cellIndex) const;
};

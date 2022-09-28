#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"

class AreaLengthObjective2 : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    VectorXd c_fixed;

    double area_target = 0.05;
    double area_weight = 2;
    double length_weight = 0.01;
    double centroid_weight = 0.01;

public:
    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &c_free) const;
};

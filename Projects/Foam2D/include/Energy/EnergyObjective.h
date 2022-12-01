#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"

class EnergyObjective : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    VectorXd c_fixed;
    VectorXd boundary;

    VectorXd area_targets = 0.05 * VectorXd::Ones(1);
    double area_weight = 0.1;
    double length_weight = 0.003;
    double centroid_weight = 0.05;
    double drag_target_weight = 0.00;

    int drag_idx = -1;
    TV drag_target_pos = {0, 0};

    int n_free; // Number of movable sites
    int n_fixed; // Number of fixed sites

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

    double getAreaTarget(int cellIndex) const;
};

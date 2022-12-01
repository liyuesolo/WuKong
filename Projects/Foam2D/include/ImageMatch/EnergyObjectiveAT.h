#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"

class EnergyObjectiveAT : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    VectorXd c_fixed;
    VectorXd boundary;

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
    getInputs(const VectorXT &c, const VectorXT &area_targets, const int cellIndex, std::vector<int> cell,
              VectorXT &p_in, VectorXT &n_in, VectorXT &c_in, VectorXT &b_in,
              VectorXi &map) const;

    virtual double evaluate(const VectorXd &x) const;

    virtual void addGradientTo(const VectorXd &x, VectorXd &grad) const;

    VectorXd get_dOdx(const VectorXd &x) const;

    virtual void getHessian(const VectorXd &x, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odx2(const VectorXd &x) const;
};

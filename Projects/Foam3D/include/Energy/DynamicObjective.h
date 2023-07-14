#pragma once

#include "EnergyObjective.h"

class DynamicObjective : public ObjectiveFunction {

public:
    EnergyObjective *energyObjective;
    bool initialized = false;

    VectorXd y_prev;
    VectorXd v_prev;

    double dt = 0.005;
    double m = 0.001;
    double eta = 0.1;

public:

    void minimize(GradientDescentLineSearch *minimizer, VectorXd &y, bool optimizeWeights_) const;

    void check_gradients(const VectorXd &y, bool optimizeWeights_) const;

    VectorXd get_a(const VectorXd &y) const;

    void newStep(const Eigen::VectorXd &y, bool optimizeWeights_);

    virtual double evaluate(const VectorXd &y) const;

    virtual void addGradientTo(const VectorXd &y, VectorXd &grad) const;

    virtual void getHessian(const VectorXd &y, SparseMatrixd &hessian) const;
};

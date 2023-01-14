#pragma once

#include "EnergyObjective.h"
#include "../Foam2DInfo.h"

class DynamicObjective : public ObjectiveFunction {

public:
    EnergyObjective *energyObjective;

    VectorXd y_prev;
    VectorXd v_prev;

    Foam2DInfo *info;

public:

    void check_gradients(const VectorXd &y) const;

    VectorXd get_a(const VectorXd &y) const;

    void newStep(const Eigen::VectorXd &y);

    virtual double evaluate(const VectorXd &y) const;

    virtual void addGradientTo(const VectorXd &y, VectorXd &grad) const;

    virtual void getHessian(const VectorXd &y, SparseMatrixd &hessian) const;
};

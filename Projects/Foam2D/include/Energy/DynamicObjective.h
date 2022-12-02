#pragma once

#include "EnergyObjective.h"
#include "../Foam2DInfo.h"

class DynamicObjective : public ObjectiveFunction {

public:
    EnergyObjective *energyObjective;

    VectorXd c_prev;
    VectorXd v_prev;

    Foam2DInfo *info;

public:

    VectorXd get_a(const VectorXd &c_free) const;

    void newStep(const Eigen::VectorXd &c_free);

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;
};

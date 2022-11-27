#pragma once

#include "EnergyObjective.h"

class DynamicObjective : public ObjectiveFunction {

public:
    EnergyObjective *energyObjective;

    VectorXd c_prev;
    VectorXd v_prev;
    VectorXd M;
    VectorXd H;
    double h;

public:

    VectorXd get_a(const VectorXd &c_free) const;

    void init(const VectorXd &c_init, double dt, double m, double mu, EnergyObjective *energy);

    void newStep(const Eigen::VectorXd &c_free);

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    virtual void getHessian(const VectorXd &c_free, SparseMatrixd &hessian) const;
};

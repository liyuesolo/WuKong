#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../src/optLib/GradientDescentMinimizer.h"
#include "CellFunctionEnergy.h"

class Tessellation;

class CellFunctionEnergy;

class EnergyObjective : public ObjectiveFunction {

public:
    mutable bool optimizeWeights = true;
    mutable int optDims = 4;
    mutable VectorXT paramsSave;

    Tessellation *tessellation;
    CellFunctionEnergy energyFunction;

public:

    void minimize(GradientDescentLineSearch *minimizer, VectorXd &y, bool optimizeWeights_) const;

    void check_gradients(const VectorXd &y, bool optimizeWeights_) const;

    void preProcess(const VectorXd &y) const;

    virtual double evaluate(const VectorXd &y) const;

    virtual void addGradientTo(const VectorXd &y, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &y) const;

    virtual void getHessian(const VectorXd &y, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odc2(const VectorXd &y) const;
};

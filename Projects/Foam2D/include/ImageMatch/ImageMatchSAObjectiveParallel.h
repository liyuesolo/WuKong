#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"
#include "../../include/ImageMatch/EnergyObjectiveAT.h"

class ImageMatchSAObjectiveParallel : public ObjectiveFunction {

public:
    std::vector<VectorXd> pix;
    double dx;
    double dy;

    VectorXT c0;

    Foam2DInfo *info;
    mutable std::map<double, VectorXT> sols;

public:
    bool preProcess(const VectorXd &tau, Tessellation *tessellation, std::vector<CellInfo> &cellInfos, VectorXd &c_free, bool need_get_c) const;

    virtual double evaluate(const VectorXd &tau) const;

    virtual void addGradientTo(const VectorXd &tau, VectorXd &grad) const;

    VectorXd get_dOdtau(const VectorXd &tau) const;

    virtual void getHessian(const VectorXd &tau, SparseMatrixd &hessian) const;

    Eigen::SparseMatrix<double> get_d2Odtau2(const VectorXd &tau) const;

    bool getC(const VectorXd &tau, Tessellation *tessellation, VectorXT &c) const;
};

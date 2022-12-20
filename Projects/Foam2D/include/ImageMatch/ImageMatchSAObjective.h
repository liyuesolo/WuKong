#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"
#include "../../include/Energy/EnergyObjective.h"
#include "../../include/ImageMatch/EnergyObjectiveAT.h"

class ImageMatchSAObjective : public ObjectiveFunction {

public:
    std::vector<VectorXd> pix;
    double dx;
    double dy;

    VectorXT c0;

    Foam2DInfo *info;
    EnergyObjectiveAT *energyObjectiveAT;

public:
    void preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const;

    virtual double evaluate(const VectorXd &tau) const;

    virtual void addGradientTo(const VectorXd &tau, VectorXd &grad) const;

    VectorXd get_dOdtau(const VectorXd &tau) const;

    bool getC(const VectorXd &tau, VectorXT &c) const;
};

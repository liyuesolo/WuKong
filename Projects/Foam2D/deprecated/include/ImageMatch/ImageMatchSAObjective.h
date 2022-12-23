#pragma once

#include "Projects/Foam2D/src/optLib/ObjectiveFunction.h"
#include "Projects/Foam2D/include/Foam2DInfo.h"
#include "Projects/Foam2D/include/ImageMatch/EnergyObjectiveAT.h"
#include "Projects/Foam2D/include/ImageMatch/CellFunctionImageMatch2AreaScaled.h"

typedef CellFunctionImageMatch2AreaScaled TypedefImageMatchFunction;

class ImageMatchSAObjective : public ObjectiveFunction {

public:
    std::vector<VectorXd> pix;
    double dx;
    double dy;

    VectorXT c0;

    Foam2DInfo *info;

public:
    void preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const;

    virtual double evaluate(const VectorXd &tau) const;

    virtual void addGradientTo(const VectorXd &tau, VectorXd &grad) const;

    VectorXd get_dOdtau(const VectorXd &tau) const;

    bool getC(const VectorXd &tau, VectorXT &c) const;
};

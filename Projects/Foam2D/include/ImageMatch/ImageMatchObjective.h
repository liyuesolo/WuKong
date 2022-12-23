#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Foam2DInfo.h"
#include "../../include/ImageMatch/ImageMatchSAObjectiveParallel.h"

class ImageMatchObjective : public ObjectiveFunction {

public:
    std::vector<VectorXd> pix;
    double dx;
    double dy;

    Foam2DInfo *info;

public:
    void preProcess(const VectorXd &c_free, std::vector<CellInfo> &cellInfos) const;

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;
};

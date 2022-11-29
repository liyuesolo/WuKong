#pragma once

#include "../../src/optLib/ObjectiveFunction.h"
#include "../../include/Tessellation/Tessellation.h"

class ImageMatchObjective : public ObjectiveFunction {

public:
    Tessellation *tessellation;

    std::vector<VectorXd> pix;

    VectorXd c_fixed;
    VectorXd boundary;

    int n_free; // Number of movable sites
    int n_fixed; // Number of fixed sites

    double dx;
    double dy;

public:
    void
    getInputs(const VectorXT &vertices, const int cellIndex, std::vector<int> cell, VectorXT &p_in,
              VectorXT &n_in, VectorXT &c_in, VectorXT &b_in, VectorXT &pix_in,
              VectorXi &map) const;

    virtual double evaluate(const VectorXd &c_free) const;

    virtual void addGradientTo(const VectorXd &c_free, VectorXd &grad) const;

    VectorXd get_dOdc(const VectorXd &c_free) const;
};

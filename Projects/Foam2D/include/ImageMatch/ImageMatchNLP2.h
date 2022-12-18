#pragma once

#include "../ImageMatch/EnergyObjectiveAT.h"
#include "../ImageMatch/ImageMatchObjective.h"

class ImageMatchNLP2 {

public:
    ImageMatchObjective *objective;
    EnergyObjectiveAT *energy;

    Foam2DInfo *info;

    VectorXd c0; // Initial positions of free sites (+ additional tessellation degrees of freedom)

    VectorXd x_guess; // Initial guess for solution.
    VectorXd x_sol; // Solution.

public:
    void check_gradients(const VectorXd &x) const;

    double eval_f(const VectorXd &x) const;

    VectorXd eval_grad_f(const VectorXd &x) const;

    VectorXd eval_g(const VectorXd &x) const;

    Eigen::SparseMatrix<double> eval_jac_g_sparsematrix(const VectorXd &x) const;
};

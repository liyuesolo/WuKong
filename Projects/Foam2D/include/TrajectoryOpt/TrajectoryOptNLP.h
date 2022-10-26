#pragma once

#include "../Objective/EnergyObjective.h"

class TrajectoryOptNLP {

public:
    EnergyObjective *energy; // Energy function (+ gradient / hessian). Includes fixed sites and tessellation info.

    int N; // Number of time steps (states from 0 to N, inclusive)
    double h; // Time step

    int agent; // Index of site whose trajectory is being optimized
    TV target_pos; // Target position for agent

    VectorXd c0; // Initial positions of free sites (+ additional tessellation degrees of freedom)
    VectorXd v0; // Initial velocities of free sites
    VectorXd x_guess;

public:
    double get_f(const VectorXd &x) const;

    VectorXd get_grad_f(const VectorXd &x) const;

    VectorXd get_g(const VectorXd &x) const;

    Eigen::SparseMatrix<double> get_jac_g(const VectorXd &x) const;
};

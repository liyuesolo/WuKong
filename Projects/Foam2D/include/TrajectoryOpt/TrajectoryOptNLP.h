#pragma once

#include "../Objective/DynamicObjective.h"

class TrajectoryOptNLP {

public:
    EnergyObjective *energy; // Energy function (+ gradient / hessian). Includes fixed sites and tessellation info.
    DynamicObjective *dynamics; // Dynamics. Used for time step and mass matrix.

    int N; // Number of time steps (states from 0 to N, inclusive)

    int agent; // Index of site whose trajectory is being optimized
    TV target_pos; // Target position for agent
    double target_weight = 1e-2;
    double velocity_weight = 1e-3;
    double input_weight = 1e-2;

    VectorXd c0; // Initial positions of free sites (+ additional tessellation degrees of freedom)
    VectorXd v0; // Initial velocities of free sites

    VectorXd x_guess; // Initial guess for solution.
    VectorXd x_sol; // Solution.

public:
    double eval_f(const VectorXd &x) const;

    VectorXd eval_grad_f(const VectorXd &x) const;

    VectorXd eval_g(const VectorXd &x) const;

    Eigen::SparseMatrix<double> eval_jac_g_sparsematrix(const VectorXd &x) const;

    std::vector<Eigen::Triplet<double>> eval_jac_g_triplets(const VectorXd &x) const;
};

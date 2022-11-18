#include <iostream>

#include "../include/Objective/EnergyObjective.h"
#include "../include/Objective/DynamicObjective.h"
#include "../include/Tessellation/Power.h"
#include "../include/TrajectoryOpt/TrajectoryOptNLP.h"

#include "inputs.h"

int main() {
    Power tessellation;

    EnergyObjective energy;
    energy.tessellation = &tessellation;
    energy.n_free = 40;
    energy.n_fixed = 40;
    energy.c_fixed = profiling_c_fixed();
    energy.drag_idx = 32;
    energy.drag_target_pos = {0, 0};

    DynamicObjective dynamics;
    dynamics.energyObjective = &energy;
    dynamics.M = 0.002 * VectorXd::Ones(energy.n_free * (2 + tessellation.getNumVertexParams()));
    dynamics.H = 0.002 * VectorXd::Ones(energy.n_free * (2 + tessellation.getNumVertexParams()));
    dynamics.h = 0.03;

    TrajectoryOptNLP nlp;
    nlp.energy = &energy;
    nlp.dynamics = &dynamics;
    nlp.N = 30;
    nlp.agent = energy.drag_idx;
    nlp.target_pos = energy.drag_target_pos;
    nlp.c0 = profiling_c0();
    nlp.v0 = VectorXd::Zero(nlp.c0.rows());

    VectorXd x = profiling_x();

    auto f = nlp.eval_f(x);
    auto grad_f = nlp.eval_grad_f(x);
    auto g = nlp.eval_g(x);
    auto jac_g_sparse_matrix = nlp.eval_jac_g_sparsematrix(x);
    auto jac_g_triplets = nlp.eval_jac_g_triplets(x);

    std::cout << "Use the results for something: "
              << f + grad_f(0) + g(0) + jac_g_sparse_matrix.coeff(0, 0) + jac_g_triplets.at(0).value() << std::endl;

    return 0;
}

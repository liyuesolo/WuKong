#include <iostream>

#include "../include/Energy/EnergyObjectiveCasadi.h"
#include "../include/Energy/EnergyObjective.h"
#include "../include/Energy/DynamicObjective.h"
#include "../include/Tessellation/Power.h"
#include "../include/Tessellation/Voronoi.h"
#include "../include/TrajectoryOpt/TrajectoryOptNLP.h"

#include "inputs.h"
#include <chrono>

//void TrajectoryOptNLPProfile() {
//    Power tessellation;
//
//    EnergyObjective energy;
//    energy.tessellation = &tessellation;
//    energy.n_free = 40;
//    energy.n_fixed = 40;
//    energy.c_fixed = profiling_c_fixed();
//    energy.drag_idx = 32;
//    energy.drag_target_pos = {0, 0};
//
//    DynamicObjective dynamics;
//    dynamics.energyObjective = &energy;
//    dynamics.M = 0.002 * VectorXd::Ones(energy.n_free * (2 + tessellation.getNumVertexParams()));
//    dynamics.H = 0.002 * VectorXd::Ones(energy.n_free * (2 + tessellation.getNumVertexParams()));
//    dynamics.h = 0.03;
//
//    TrajectoryOptNLP nlp;
//    nlp.energy = &energy;
//    nlp.dynamics = &dynamics;
//    nlp.N = 30;
//    nlp.agent = energy.drag_idx;
//    nlp.target_pos = energy.drag_target_pos;
//    nlp.c0 = profiling_c0();
//    nlp.v0 = VectorXd::Zero(nlp.c0.rows());
//
//    VectorXd x = profiling_x();
//
//    auto f = nlp.eval_f(x);
//    auto grad_f = nlp.eval_grad_f(x);
//    auto g = nlp.eval_g(x);
//    auto jac_g_sparse_matrix = nlp.eval_jac_g_sparsematrix(x);
//    auto jac_g_triplets = nlp.eval_jac_g_triplets(x);
//
//    std::cout << "Use the results for something: "
//              << f + grad_f(0) + g(0) + jac_g_sparse_matrix.coeff(0, 0) + jac_g_triplets.at(0).value() << std::endl;
//
//    return 0;
//}

void EnergyObjectiveProfile() {
    Foam2DInfo info;

    Voronoi tessellation;
    info.tessellations.push_back(&tessellation);

    info.n_free = 40;
    info.n_fixed = 8;

    double dx = 0.75;
    double dy = 0.75;
    info.boundary.resize(4 * 2);
    info.boundary << -dx, -dy, dx, -dy, dx, dy, -dx, dy;

    int dims = 2;
    VectorXT c = energy_profiling_c();
    info.c_fixed = c.segment(info.n_free * dims, info.n_fixed * dims);
    VectorXT c_free = c.segment(0, info.n_free * dims);

    info.energy_area_targets = 0.05 * VectorXT::Ones(info.n_free);

    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::milliseconds ms;
    typedef std::chrono::duration<float> fsec;

    EnergyObjective energy2;
    energy2.info = &info;
    energy2.evaluate(c_free);
    energy2.get_dOdc(c_free);
    std::cout << "Start hess2" << std::endl;
    auto start = Time::now();
    for (int i = 0; i < 500; i++) {
        energy2.get_d2Odc2(c_free);
        info.getTessellation()->c(0) = 0;
    }
    auto end = Time::now();
    ms d = std::chrono::duration_cast<ms>(end - start);
    std::cout << d.count() << " ms\n";

    EnergyObjectiveCasadi energy1;
    energy1.info = &info;
    energy1.evaluate(c_free);
    energy1.get_dOdc(c_free);
    std::cout << "Start hess1" << std::endl;
    start = Time::now();
    for (int i = 0; i < 500; i++) {
        energy1.get_d2Odc2(c_free);
    }
    end = Time::now();
    d = std::chrono::duration_cast<ms>(end - start);
    std::cout << d.count() << " ms\n";
}

int main() {
    EnergyObjectiveProfile();

    return 0;
}

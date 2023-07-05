#include "../include/Foam3D.h"
#include "Projects/Foam3D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"
#include <random>
#include <thread>
#include "../src/optLib/ParallelLineSearchMinimizers.h"

Foam3D::Foam3D() {
    energyObjective.tessellation = &tessellation;

    minimizerGradientDescent = new GradientDescentLineSearch(1, 1e-6, 15);
    minimizerNewton = new NewtonFunctionMinimizer(1, 1e-10, 15);
    minimizerBFGS = new FancyBFGSMinimizer(1, 1e-10, 15);
}

void Foam3D::energyMinimizationStep(int optimizer) {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y = c.segment(0, c.rows() - 8 * 4);

    bool optWeights = false;
//    energyObjective.check_gradients(y, optWeights);
    switch (optimizer) {
        case 0:
            energyObjective.minimize(minimizerGradientDescent, y, optWeights);
            break;
        case 1:
            energyObjective.minimize(minimizerNewton, y, optWeights);
            break;
        case 2:
            energyObjective.minimize(minimizerBFGS, y, optWeights);
            break;
        default:
            break;
    }

    double infp = 10;
    VectorXd infbox(8 * 4);
    infbox << -infp, -infp, -infp, 0,
            -infp, -infp, infp, 0,
            -infp, infp, -infp, 0,
            -infp, infp, infp, 0,
            infp, -infp, -infp, 0,
            infp, -infp, infp, 0,
            infp, infp, -infp, 0,
            infp, infp, infp, 0;
    VectorXd y_with_infbox(y.rows() + infbox.rows());
    y_with_infbox << y, infbox;
    tessellation.separateVerticesParams(y_with_infbox, vertices, params);
}

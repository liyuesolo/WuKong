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
    VectorXT y(c.rows() - 8 * 4 + tessellation.boundary->nfree);
    y << c.segment(0, c.rows() - 8 * 4), tessellation.boundary->get_p_free();

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

    c.segment(0, c.rows() - 8 * 4) = y.segment(0, c.rows() - 8 * 4);
    VectorXT p_free = y.tail(tessellation.boundary->nfree);

    tessellation.separateVerticesParams(c, vertices, params);
    tessellation.tessellate(vertices, params, p_free);
}

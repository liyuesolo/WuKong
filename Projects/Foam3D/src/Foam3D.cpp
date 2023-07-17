#include "../include/Foam3D.h"
#include "Projects/Foam3D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"
#include <random>
#include <thread>
#include "../src/optLib/ParallelLineSearchMinimizers.h"

Foam3D::Foam3D() {
    energyObjective.tessellation = &tessellation;
    dynamicObjective.energyObjective = &energyObjective;

    minimizerGradientDescent = new GradientDescentLineSearch(1, 1e-6, 15);
    minimizerNewton = new NewtonFunctionMinimizer(1, 1e-2, 15);
    minimizerBFGS = new FancyBFGSMinimizer(1, 1e-10, 15);
}

void Foam3D::energyMinimizationStep(int optimizer) {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y(c.rows() + tessellation.boundary->nfree);
    y << c, tessellation.boundary->get_p_free();

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

    c = y.segment(0, c.rows());
    VectorXT p_free = y.tail(tessellation.boundary->nfree);

    tessellation.separateVerticesParams(c, vertices, params);
    tessellation.tessellate(vertices, params, p_free);
}

void Foam3D::dynamicsStep(int optimizer) {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y(c.rows() + tessellation.boundary->nfree);
    y << c, tessellation.boundary->get_p_free();
    VectorXT y_prev = y;

    bool optWeights = false;
//    dynamicObjective.check_gradients(y, optWeights);
    switch (optimizer) {
        case 0:
            dynamicObjective.minimize(minimizerGradientDescent, y, optWeights);
            break;
        case 1:
            dynamicObjective.minimize(minimizerNewton, y, optWeights);
            break;
        case 2:
            dynamicObjective.minimize(minimizerBFGS, y, optWeights);
            break;
        default:
            break;
    }

    c = y.segment(0, c.rows());
    VectorXT p_free = y.tail(tessellation.boundary->nfree);

    tessellation.separateVerticesParams(c, vertices, params);
    tessellation.tessellate(vertices, params, p_free);

    if ((y - y_prev).norm() < 1e-14) {
        std::cout << std::endl << "New dynamics step" << std::endl << std::endl;
        dynamicObjective.newStep(y, optWeights);
        minimizerGradientDescent->alpha_start = 1;
        minimizerNewton->alpha_start = 1;
        minimizerBFGS->alpha_start = 1;
    }
}

void Foam3D::dynamicsInit() {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y(c.rows() + tessellation.boundary->nfree);
    y << c, tessellation.boundary->get_p_free();

    bool optWeights = false;
    dynamicObjective.newStep(y, optWeights);
}

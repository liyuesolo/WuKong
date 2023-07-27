#include "../include/Foam3D.h"
#include "Projects/Foam3D/include/Tessellation/Power.h"
#include <random>
#include <thread>
#include "../include/Energy/EnergyMinimizer.h"

Foam3D::Foam3D() {
    energyObjective.tessellation = &tessellation;
    dynamicObjective.energyObjective = &energyObjective;

    minimizerGradientDescent = new GradientDescentLineSearch(1, 1e-6, 15);
    minimizerNewton = new NewtonFunctionMinimizer(1, 1e-2, 15);
    minimizerBFGS = new FancyBFGSMinimizer(1, 1e-10, 15);

    energyMinimizer.tessellation = &tessellation;
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
            energyMinimizer.advanceOneStep(0, y, false, optWeights);
//            energyObjective.minimize(minimizerNewton, y, optWeights);
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
    tessellation.tessellate(vertices, params, p_free, true);
}

bool Foam3D::dynamicsStep(int optimizer) {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y(c.rows() + tessellation.boundary->nfree);
    y << c, tessellation.boundary->get_p_free();

    bool optWeights = false;
//    dynamicObjective.check_gradients(y, optWeights);
    bool converged = false;
    switch (optimizer) {
        case 0:
            dynamicObjective.minimize(minimizerGradientDescent, y, optWeights);
            break;
        case 1:
            converged = energyMinimizer.advanceOneStep(0, y, true, optWeights);
//            dynamicObjective.minimize(minimizerNewton, y, optWeights);
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
    tessellation.tessellate(vertices, params, p_free, true);

    if (converged) {
        std::cout << std::endl << "New dynamics step" << std::endl << std::endl;
        dynamicObjective.newStep(y, optWeights);
        minimizerGradientDescent->alpha_start = 1;
        minimizerNewton->alpha_start = 1;
        minimizerBFGS->alpha_start = 1;

        energyMinimizer.dynamic_new_step = true;

        return true;
    } else {
        return false;
    }
}

void Foam3D::dynamicsInit() {
    VectorXT c = tessellation.combineVerticesParams(vertices, params);
    VectorXT y(c.rows() + tessellation.boundary->nfree);
    y << c, tessellation.boundary->get_p_free();

    bool optWeights = false;
    dynamicObjective.newStep(y, optWeights);

    energyMinimizer.dynamic_initialized = false;
}

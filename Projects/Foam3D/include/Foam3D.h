#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"
#include "Tessellation/Power.h"
#include "Energy/EnergyObjective.h"

#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"

class Foam3D {
public:
    VectorXT vertices;
    VectorXT params;

    Power tessellation;
    EnergyObjective energyObjective;

    GradientDescentLineSearch *minimizerGradientDescent;
    NewtonFunctionMinimizer *minimizerNewton;
    FancyBFGSMinimizer *minimizerBFGS;

public:
    void energyMinimizationStep(int optimizer);

public:

    Foam3D();

    ~Foam3D() {}
};

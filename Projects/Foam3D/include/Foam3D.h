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
#include "Energy/DynamicObjective.h"

#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"

class Foam3D {
public:
    VectorXT vertices;
    VectorXT params;

    Power tessellation;
    EnergyObjective energyObjective;
    DynamicObjective dynamicObjective;

    GradientDescentLineSearch *minimizerGradientDescent;
    NewtonFunctionMinimizer *minimizerNewton;
    FancyBFGSMinimizer *minimizerBFGS;

public:
    void energyMinimizationStep(int optimizer);

    void dynamicsStep(int optimizer);

    void dynamicsInit();

public:

    Foam3D();

    ~Foam3D() {}
};

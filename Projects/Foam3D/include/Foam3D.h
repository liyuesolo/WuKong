#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>

#include "VecMatDef.h"
#include "../src/optLib/GradientDescentMinimizer.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"
#include "../src/optLib/ParallelLineSearchMinimizers.h"
#include "Tessellation/Power.h"

class Foam3D {
public:
    VectorXT vertices;
    VectorXT params;

    Power tessellation;

public:

public:

    Foam3D();

    ~Foam3D() {}
};

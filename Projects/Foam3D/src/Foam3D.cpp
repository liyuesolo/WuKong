#include <igl/triangle/triangulate.h>

#include "../include/Foam3D.h"
#include "Projects/Foam3D/include/Tessellation/Power.h"
#include "../src/optLib/NewtonFunctionMinimizer.h"
#include "../src/optLib/FancyBFGSMinimizer.h"
#include <random>
#include <thread>
#include "../src/optLib/ParallelLineSearchMinimizers.h"

Foam3D::Foam3D() {

}

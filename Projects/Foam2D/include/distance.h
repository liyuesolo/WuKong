#include <eigen3/Eigen/Dense>
#include <functional>
#include <cmath>
#include "ad.h"

enum VertexEdgeMode {K, L, Edge, Null};  // how to compute the distance

VertexEdgeMode vertexEdgeMode(const std::array<double, 7> x);

double vertexToEdgeDistance(const std::array<double,7> x);
double vertexToEdgeDistancePotential(const std::array<double,7> x);
double vertexToLineDistancePotential(const std::array<double,7> x);
bool vertexToEdgeDistanceValid(const std::array<double,7>& x);

void vertexToLineDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x);
void vertexToEdgeDistancePotentialJacobian(std::array<double, 6>& jacobian, const std::array<double, 7>& x);

void vertexToEdgeDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x);
void vertexToLineDistancePotentialHessian(std::array<double, 36>& hessian, const std::array<double, 7>& x);

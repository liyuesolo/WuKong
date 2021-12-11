#ifndef ELASTICITY_H
#define ELASTICITY_H
#include <iostream>

#include "../VecMatDef.h"

void compute3DNeoHookeanEnergyEnu(double E, double nu, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & xUndef, double& energy);
void compute3DNeoHookeanEnergyEnuGradient(double E, double nu, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & xUndef, Eigen::Matrix<double, 12, 1>& energygradient);
void compute3DNeoHookeanEnergyEnuHessian(double E, double nu, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & xUndef, Eigen::Matrix<double, 12, 12>& energyhessian);
#endif
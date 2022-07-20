#ifndef FEM_ENERGY_H
#define FEM_ENERGY_H

#include "../VecMatDef.h"

void computeLinear2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double& energy);
void computeLinear2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 1>& energygradient);
void computeLinear2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& energyhessian);

#endif
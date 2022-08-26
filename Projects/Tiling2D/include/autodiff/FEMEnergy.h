#ifndef FEM_ENERGY_H
#define FEM_ENERGY_H

#include "../VecMatDef.h"

void computeLinear2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double& energy);
void computeLinear2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 1>& energygradient);
void computeLinear2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& energyhessian);
void computeLinear2DNeoHookeandfdX(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& dfdX);

void computeQuadratic2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, double& energy);
void computeQuadratic2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 1>& energygradient);
void computeQuadratic2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 12>& energyhessian);
void computeQuadratic2DNeoHookeandfdX(double E, double nu, const Eigen::Matrix<double,6,2> & x, const Eigen::Matrix<double,6,2> & X, Eigen::Matrix<double, 12, 12>& dfdX);
#endif
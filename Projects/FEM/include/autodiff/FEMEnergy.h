#ifndef FEM_ENERGY_H
#define FEM_ENERGY_H

#include "../VecMatDef.h"

void computeLinearTet3DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, double& energy);
void computeLinearTet3DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, Eigen::Matrix<double, 12, 1>& energygradient);
void computeLinearTet3DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,4,3> & x, const Eigen::Matrix<double,4,3> & X, Eigen::Matrix<double, 12, 12>& energyhessian);

void neoHookeandPdF(double lambda, double mu, const Eigen::Matrix<double,3,3> & F, Eigen::Matrix<double, 9, 9>& dPdF);

void computeLinear2DNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, double& energy);
void computeLinear2DNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 1>& energygradient);
void computeLinear2DNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,3,2> & x, const Eigen::Matrix<double,3,2> & X, Eigen::Matrix<double, 6, 6>& energyhessian);

#endif
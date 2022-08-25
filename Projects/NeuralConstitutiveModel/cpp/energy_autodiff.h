#ifndef ENERGY_AUTODIFF_H
#define ENERGY_AUTODIFF_H

#include <Eigen/Core>
#include <iostream>

void compute2DOrthoStVkEnergy(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,4,1> & ExEy_nuxy_nuyx, double& energy);
void compute2DOrthoStVkEnergyGradient(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,4,1> & ExEy_nuxy_nuyx, Eigen::Matrix<double, 3, 1>& energygradient);
void compute2DOrthoStVkEnergyHessian(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,4,1> & ExEy_nuxy_nuyx, Eigen::Matrix<double, 3, 3>& energyhessian);

void compute2DisoStVkEnergy(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,2,1> & Enu, double& energy);
void compute2DisoStVkEnergyGradient(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,2,1> & Enu, Eigen::Matrix<double, 3, 1>& energygradient);
void compute2DisoStVkEnergyHessian(double lambda1, double lambda2, double theta, const Eigen::Matrix<double,2,1> & Enu, Eigen::Matrix<double, 3, 3>& energyhessian);

#endif
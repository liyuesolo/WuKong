#ifndef PBC_ENERGY_H
#define PBC_ENERGY_H

#include "../VecMatDef.h"

void computeStrainMatchingEnergy(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, double& energy);
void computeStrainMatchingEnergyGradient(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 1>& energygradient);
void computeStrainMatchingEnergyHessian(double stiffness, const Eigen::Matrix<double,3,1> & epsilon, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 8>& energyhessian);

#endif
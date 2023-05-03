#ifndef REPULSION_H
#define REPULSION_H

#include "../include/VecMatDef.h"

void computeRepulsion2DCubicEnergy(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double dhat, double& energy);
void computeRepulsion2DCubicEnergyGradient(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double dhat, Eigen::Matrix<double, 4, 1>& energygradient);
void computeRepulsion2DCubicEnergyHessian(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double dhat, Eigen::Matrix<double, 4, 4>& energyhessian);


void computeRepulsion3DCubicEnergy(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double dhat, double& energy);
void computeRepulsion3DCubicEnergyGradient(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double dhat, Eigen::Matrix<double, 6, 1>& energygradient);
void computeRepulsion3DCubicEnergyHessian(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double dhat, Eigen::Matrix<double, 6, 6>& energyhessian);
#endif
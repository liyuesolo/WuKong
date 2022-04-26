#ifndef MEMBRANE_ENERGY_H
#define MEMBRANE_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void computeMembraneQubicPenalty(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, double& energy);
void computeMembraneQubicPenaltyGradient(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 2, 1>& energygradient);
void computeMembraneQubicPenaltyHessian(double stiffness, double radius, const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 2, 2>& energyhessian);

#endif
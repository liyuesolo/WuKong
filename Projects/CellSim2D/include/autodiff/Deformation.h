#ifndef DEFORMATION_H
#define DEFORMATION_H

#include <iostream>

#include "../VecMatDef.h"

void computeDeformationPenalty(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, double& energy);
void computeDeformationPenaltyGradient(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 1>& energygradient);
void computeDeformationPenaltyHessian(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 6>& energyhessian);


void computeDeformationPenaltyDet(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, double& energy);
void computeDeformationPenaltyDetGradient(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 1>& energygradient);
void computeDeformationPenaltyDetHessian(double stiffness, const Eigen::Matrix<double,6,1> & X, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 6>& energyhessian);

#endif
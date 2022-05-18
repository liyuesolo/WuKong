#ifndef DEFORMATION_H
#define DEFORMATION_H

#include <iostream>

#include "../VecMatDef.h"

void computeDeformationPenaltyDet3D(double stiffness, const Eigen::Matrix<double,12,1> & X, const Eigen::Matrix<double,12,1> & x, double& energy);
void computeDeformationPenaltyDet3DGradient(double stiffness, const Eigen::Matrix<double,12,1> & X, const Eigen::Matrix<double,12,1> & x, Eigen::Matrix<double, 12, 1>& energygradient);
void computeDeformationPenaltyDet3DHessian(double stiffness, const Eigen::Matrix<double,12,1> & X, const Eigen::Matrix<double,12,1> & x, Eigen::Matrix<double, 12, 12>& energyhessian);

void computeDeformationPenaltyCST3D(double stiffness, const Eigen::Matrix<double,9,1> & X, const Eigen::Matrix<double,9,1> & x, double& energy);
void computeDeformationPenaltyCST3DGradient(double stiffness, const Eigen::Matrix<double,9,1> & X, const Eigen::Matrix<double,9,1> & x, Eigen::Matrix<double, 9, 1>& energygradient);
void computeDeformationPenaltyCST3DHessian(double stiffness, const Eigen::Matrix<double,9,1> & X, const Eigen::Matrix<double,9,1> & x, Eigen::Matrix<double, 9, 9>& energyhessian);

#endif
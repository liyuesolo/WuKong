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


void computeSpringUnilateralQubicEnergy2D(double stiffness, const Eigen::Matrix<double,4,1> & X, const Eigen::Matrix<double,4,1> & x, double& energy);
void computeSpringUnilateralQubicEnergy2DGradient(double stiffness, const Eigen::Matrix<double,4,1> & X, const Eigen::Matrix<double,4,1> & x, Eigen::Matrix<double, 4, 1>& energygradient);
void computeSpringUnilateralQubicEnergy2DHessian(double stiffness, const Eigen::Matrix<double,4,1> & X, const Eigen::Matrix<double,4,1> & x, Eigen::Matrix<double, 4, 4>& energyhessian);


void computeSpringUnilateralQubicEnergyRestLength2D(double stiffness, double l0, const Eigen::Matrix<double,4,1> & x, double& energy);
void computeSpringUnilateralQubicEnergyRestLength2DGradient(double stiffness, double l0, const Eigen::Matrix<double,4,1> & x, Eigen::Matrix<double, 4, 1>& energygradient);
void computeSpringUnilateralQubicEnergyRestLength2DHessian(double stiffness, double l0, const Eigen::Matrix<double,4,1> & x, Eigen::Matrix<double, 4, 4>& energyhessian);

#endif
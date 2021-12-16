#ifndef TET_VOL_PENALTY_H
#define TET_VOL_PENALTY_H


#include <iostream>

#include "../VecMatDef.h"

void computeSingleTetVolPenalty(double stiffness, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & X, double& energy);
void computeSingleTetVolPenaltyGradient(double stiffness, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & X, Eigen::Matrix<double, 12, 1>& energygradient);
void computeSingleTetVolPenaltyHessian(double stiffness, const Eigen::Matrix<double,3,4> & x, const Eigen::Matrix<double,3,4> & X, Eigen::Matrix<double, 12, 12>& energyhessian);


void computeSingleTetVol(const Eigen::Matrix<double,3,4> & x, double& energy);
void computeSingleTetVolGradient(const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 12, 1>& energygradient);
void computeSingleTetVolHessian(const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 12, 12>& energyhessian);

#endif
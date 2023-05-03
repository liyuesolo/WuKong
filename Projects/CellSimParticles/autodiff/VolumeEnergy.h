#ifndef VOLUME_ENERGY_H
#define VOLUME_ENERGY_H

#include <iostream>

#include "../include/VecMatDef.h"

void computeTetVolume(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, double& energy);
void computeTetVolumeGradient(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 1>& energygradient);
void computeTetVolumeHessian(const Eigen::Matrix<double,3,1> & v0, const Eigen::Matrix<double,3,1> & v1, const Eigen::Matrix<double,3,1> & v2, const Eigen::Matrix<double,3,1> & v3, Eigen::Matrix<double, 9, 9>& energyhessian);

#endif
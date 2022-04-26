#ifndef EDGE_ENERGY_H
#define EDGE_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void computeEdgeSquaredNorm2D(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, double& energy);
void computeEdgeSquaredNorm2DGradient(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, Eigen::Matrix<double, 4, 1>& energygradient);
void computeEdgeSquaredNorm2DHessian(const Eigen::Matrix<double,2,1> & r0, const Eigen::Matrix<double,2,1> & r1, Eigen::Matrix<double, 4, 4>& energyhessian);


#endif
#ifndef AREA_ENERGY_H
#define AREA_ENERGY_H

#include <iostream>

#include "../include/VecMatDef.h"

void computeSignedTriangleArea(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, double& energy);
void computeSignedTriangleAreaGradient(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 1>& energygradient);
void computeSignedTriangleAreaHessian(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 4>& energyhessian);



#endif
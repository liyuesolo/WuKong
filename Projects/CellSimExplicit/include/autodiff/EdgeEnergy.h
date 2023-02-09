#ifndef EDGE_ENERGY_H
#define EDGE_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

double computeEdgeLength(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1);
void computeEdgeLengthGradient(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, Matrix<double, 6, 1>& energygradient);
void computeEdgeLengthHessian(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, Matrix<double, 6, 6>& energyhessian);

double computeEdgeSquaredNorm(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1);
void computeEdgeSquaredNormGradient(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, Matrix<double, 6, 1>& energygradient);
void computeEdgeSquaredNormHessian(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, Matrix<double, 6, 6>& energyhessian);


void computeEdgeEnergyRestLength3D(double stiffness, double l0, const Eigen::Matrix<double,6,1> & x, double& energy);
void computeEdgeEnergyRestLength3DGradient(double stiffness, double l0, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 1>& energygradient);
void computeEdgeEnergyRestLength3DHessian(double stiffness, double l0, const Eigen::Matrix<double,6,1> & x, Eigen::Matrix<double, 6, 6>& energyhessian);
#endif
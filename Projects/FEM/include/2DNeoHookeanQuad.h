#ifndef _COMPUTE2DQUADNEOHOOKEANENERGY_H
#define _COMPUTE2DQUADNEOHOOKEANENERGY_H

#include <Eigen/Core>

void compute2DQuadNeoHookeanEnergy(double E, double nu, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, double& energy);
void compute2DQuadNeoHookeanEnergyGradient(double E, double nu, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 1>& energygradient);
void compute2DQuadNeoHookeanEnergyHessian(double E, double nu, const Eigen::Matrix<double,4,2> & x, const Eigen::Matrix<double,4,2> & X, Eigen::Matrix<double, 8, 8>& energyhessian);
#endif
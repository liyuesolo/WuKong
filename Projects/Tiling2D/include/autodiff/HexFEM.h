#ifndef HEXFEM_H
#define HEXFEM_H

#include "../VecMatDef.h"

#include <iostream>
void computeHexFEMPlainStrainStVKEnergy(double stiffness, double stiffness_shear, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, 
	const Eigen::Matrix<double,3,8> & xUndef, double& energy);
void computeHexFEMPlainStrainStVKEnergyGradient(double stiffness, double stiffness_shear, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, 
	const Eigen::Matrix<double,3,8> & xUndef, Eigen::Matrix<double, 24, 1>& energygradient);
void computeHexFEMPlainStrainStVKEnergyHessian(double stiffness, double stiffness_shear, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, 
	const Eigen::Matrix<double,3,8> & xUndef, Eigen::Matrix<double, 24, 24>& energyhessian);

void computeHexFEMPlainStrainNHEnergy(double stiffness, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, const Eigen::Matrix<double,3,8> & xUndef, 
	double& energy);
void computeHexFEMPlainStrainNHEnergyGradient(double stiffness, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, const Eigen::Matrix<double,3,8> & xUndef, 
	Eigen::Matrix<double, 24, 1>& energygradient);
void computeHexFEMPlainStrainNHEnergyHessian(double stiffness, double lambda, double mu, const Eigen::Matrix<double,3,8> & x, const Eigen::Matrix<double,3,8> & xUndef, 
	Eigen::Matrix<double, 24, 24>& energyhessian);
	
T computeHexFEMNeoHookeanEnergy(T lambda, T mu, const Matrix<T,3,8> & x, const Matrix<T,3,8> & xUndef);
void computeHexFEMNeoHookeanEnergyGradient(T lambda, T mu, const Matrix<T,3,8> & x, const Matrix<T,3,8> & xUndef, Matrix<T, 24, 1>& energygradient);

void computeHexFEMNeoHookeanEnergyHessian(T lambda, T mu, const Matrix<T,3,8> & x, const Matrix<T,3,8> & xUndef, Matrix<T, 24, 24>& energyhessian);

double computeHexFEMStVKEnergy(double lambda, double mu, const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef);
void computeHexFEMStVKEnergyGradient(double lambda, double mu, const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef, Matrix<double, 24, 1>& energygradient);
void computeHexFEMStVKEnergyHessian(double lambda, double mu, const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef, Matrix<double, 24, 24>& energyhessian);
double computeHexFEMVolume(const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef);
void computeHexFEMVolumeGradient(const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef, Matrix<double, 24, 1>& energygradient);
void computeHexFEMVolumeHessian(const Matrix<double,3,8> & x, const Matrix<double,3,8> & xUndef, Matrix<double, 24, 24>& energyhessian);
#endif
#ifndef EDGE_ENERGY_H
#define EDGE_ENERGY_H

#include <iostream>

#include "../include/VecMatDef.h"

void compute3DMatchingEnergy(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, double& energy);
void compute3DMatchingEnergyGradient(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, Eigen::Matrix<double, 3, 1>& energygradient);
void compute3DMatchingEnergyHessian(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, Eigen::Matrix<double, 3, 3>& energyhessian);

void computeEdgeSpringEnergy2D(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double rest_length, double& energy);
void computeEdgeSpringEnergy2DGradient(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double rest_length, Eigen::Matrix<double, 4, 1>& energygradient);
void computeEdgeSpringEnergy2DHessian(const Eigen::Matrix<double,2,1> & xi, const Eigen::Matrix<double,2,1> & xj, double rest_length, Eigen::Matrix<double, 4, 4>& energyhessian);
void computeEdgeSpringEnergy3D(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, double& energy);
void computeEdgeSpringEnergy3DGradient(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, Eigen::Matrix<double, 6, 1>& energygradient);
void computeEdgeSpringEnergy3DHessian(const Eigen::Matrix<double,3,1> & xi, const Eigen::Matrix<double,3,1> & xj, double rest_length, Eigen::Matrix<double, 6, 6>& energyhessian);

template <int dim>
void computeEdgeSpringEnergy(const Eigen::Matrix<double,dim,1> & xi, const Eigen::Matrix<double,dim,1> & xj, double rest_length, double& energy)
{
    if constexpr (dim == 2)
        computeEdgeSpringEnergy2D(xi, xj, rest_length, energy);
    else
        computeEdgeSpringEnergy3D(xi, xj, rest_length, energy);
}
template <int dim>
void computeEdgeSpringEnergyGradient(const Eigen::Matrix<double,dim,1> & xi, const Eigen::Matrix<double,dim,1> & xj, double rest_length, Eigen::Matrix<double, dim * 2, 1>& energygradient)
{
    if constexpr (dim == 2)
        computeEdgeSpringEnergy2DGradient(xi, xj, rest_length, energygradient);
    else
        computeEdgeSpringEnergy3DGradient(xi, xj, rest_length, energygradient);
}

template <int dim>
void computeEdgeSpringEnergyHessian(const Eigen::Matrix<double,dim,1> & xi, const Eigen::Matrix<double,dim,1> & xj, double rest_length, Eigen::Matrix<double, dim * 2, dim * 2>& energyhessian)
{
    if constexpr (dim == 2)
        computeEdgeSpringEnergy2DHessian(xi, xj, rest_length, energyhessian);
    else
        computeEdgeSpringEnergy3DHessian(xi, xj, rest_length, energyhessian);
}

#endif
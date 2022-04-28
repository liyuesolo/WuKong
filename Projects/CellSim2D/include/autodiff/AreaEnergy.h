#ifndef AREA_ENERGY_H
#define AREA_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void computeArea4PointsSquared2D(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, const Eigen::Matrix<double,8,1> & face_vertices_rest, double& energy);
void computeArea4PointsSquared2DGradient(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, const Eigen::Matrix<double,8,1> & face_vertices_rest, Eigen::Matrix<double, 8, 1>& energygradient);
void computeArea4PointsSquared2DHessian(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, const Eigen::Matrix<double,8,1> & face_vertices_rest, Eigen::Matrix<double, 8, 8>& energyhessian);

void computeTriangleArea(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, double& energy);
void computeTriangleAreaGradient(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 1>& energygradient);
void computeTriangleAreaHessian(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 4>& energyhessian);


void computeSignedTriangleArea(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, double& energy);
void computeSignedTriangleAreaGradient(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 1>& energygradient);
void computeSignedTriangleAreaHessian(const Eigen::Matrix<double,2,1> & vi, const Eigen::Matrix<double,2,1> & vj, const Eigen::Matrix<double,2,1> & center, Eigen::Matrix<double, 4, 4>& energyhessian);

void computeAreaBarrier4Points(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, double& energy);
void computeAreaBarrier4PointsGradient(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, Eigen::Matrix<double, 8, 1>& energygradient);
void computeAreaBarrier4PointsHessian(double stiffness, const Eigen::Matrix<double,8,1> & face_vertices, Eigen::Matrix<double, 8, 8>& energyhessian);

#endif
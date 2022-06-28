#ifndef TET_VOL_BARRIER_SCALED_H
#define TET_VOL_BARRIER_SCALED_H


#include <iostream>

#include "../VecMatDef.h"

void computeVolumeBarrier4PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,24,1> & cell_vertices, double& energy);
void computeVolumeBarrier4PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,24,1> & cell_vertices, Eigen::Matrix<double, 24, 1>& energygradient);
void computeVolumeBarrier4PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,24,1> & cell_vertices, Eigen::Matrix<double, 24, 24>& energyhessian);
void computeVolumeBarrier5PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,30,1> & cell_vertices, double& energy);
void computeVolumeBarrier5PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolumeBarrier5PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 30>& energyhessian);
void computeVolumeBarrier6PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,36,1> & cell_vertices, double& energy);
void computeVolumeBarrier6PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolumeBarrier6PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 36>& energyhessian);
void computeVolumeBarrier7PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,42,1> & cell_vertices, double& energy);
void computeVolumeBarrier7PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 1>& energygradient);
void computeVolumeBarrier7PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 42>& energyhessian);
void computeVolumeBarrier8PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,48,1> & cell_vertices, double& energy);
void computeVolumeBarrier8PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 1>& energygradient);
void computeVolumeBarrier8PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 48>& energyhessian);
void computeVolumeBarrier9PointsScaled(double stiffness, double scaling_factor, const Eigen::Matrix<double,54,1> & cell_vertices, double& energy);
void computeVolumeBarrier9PointsScaledGradient(double stiffness, double scaling_factor, const Eigen::Matrix<double,54,1> & cell_vertices, Eigen::Matrix<double, 54, 1>& energygradient);
void computeVolumeBarrier9PointsScaledHessian(double stiffness, double scaling_factor, const Eigen::Matrix<double,54,1> & cell_vertices, Eigen::Matrix<double, 54, 54>& energyhessian);



#endif
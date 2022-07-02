#ifndef YOLK_ENERGY_H
#define YOLK_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void computeConeVolume4Points(const Matrix<double,12,1> & face_vertices, const Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume4PointsGradient(const Matrix<double,12,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 12, 1>& energygradient);
void computeConeVolume4PointsHessian(const Matrix<double,12,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 12, 12>& energyhessian);

void computeConeVolume5Points(const Matrix<double,15,1> & face_vertices, const Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume5PointsGradient(const Matrix<double,15,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 15, 1>& energygradient);
void computeConeVolume5PointsHessian(const Matrix<double,15,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 15, 15>& energyhessian);

void computeConeVolume6Points(const Matrix<double,18,1> & face_vertices, const Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume6PointsGradient(const Matrix<double,18,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 18, 1>& energygradient);
void computeConeVolume6PointsHessian(const Matrix<double,18,1> & face_vertices, const Matrix<double,3,1> & centroid, Matrix<double, 18, 18>& energyhessian);

void computeConeVolume7Points(const Eigen::Matrix<double,21,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume7PointsGradient(const Eigen::Matrix<double,21,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 21, 1>& energygradient);
void computeConeVolume7PointsHessian(const Eigen::Matrix<double,21,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 21, 21>& energyhessian);

void computeConeVolume8Points(const Eigen::Matrix<double,24,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume8PointsGradient(const Eigen::Matrix<double,24,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 24, 1>& energygradient);
void computeConeVolume8PointsHessian(const Eigen::Matrix<double,24,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 24, 24>& energyhessian);

void computeQuadConeVolume(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeQuadConeVolumeGradient(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 12, 1>& energygradient);
void computeQuadConeVolumeHessian(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 12, 12>& energyhessian);

void computePentaConeVolume(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computePentaConeVolumeGradient(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 15, 1>& energygradient);
void computePentaConeVolumeHessian(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 15, 15>& energyhessian);

void computeHexConeVolume(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeHexConeVolumeGradient(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 18, 1>& energygradient);
void computeHexConeVolumeHessian(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 18, 18>& energyhessian);

void computeSepConeVolume(const Eigen::Matrix<double,21,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeSepConeVolumeGradient(const Eigen::Matrix<double,21,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 21, 1>& energygradient);
void computeSepConeVolumeHessian(const Eigen::Matrix<double,21,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 21, 21>& energyhessian);

void computeOctConeVolume(const Eigen::Matrix<double,24,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeOctConeVolumeGradient(const Eigen::Matrix<double,24,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 24, 1>& energygradient);
void computeOctConeVolumeHessian(const Eigen::Matrix<double,24,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 24, 24>& energyhessian);

void computeConeVolume9Points(const Eigen::Matrix<double,27,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, double& energy);
void computeConeVolume9PointsGradient(const Eigen::Matrix<double,27,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 27, 1>& energygradient);
void computeConeVolume9PointsHessian(const Eigen::Matrix<double,27,1> & face_vertices, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 27, 27>& energyhessian);

#endif
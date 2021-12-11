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


void computeQuadConeVolume(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeQuadConeVolumeGradient(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 12, 1>& energygradient);
void computeQuadConeVolumeHessian(const Eigen::Matrix<double,12,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 12, 12>& energyhessian);

void computePentaConeVolume(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computePentaConeVolumeGradient(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 15, 1>& energygradient);
void computePentaConeVolumeHessian(const Eigen::Matrix<double,15,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 15, 15>& energyhessian);

void computeHexConeVolume(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, double& energy);
void computeHexConeVolumeGradient(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 18, 1>& energygradient);
void computeHexConeVolumeHessian(const Eigen::Matrix<double,18,1> & vertices, const Eigen::Matrix<double,3,1> & reference_point, Eigen::Matrix<double, 18, 18>& energyhessian);


#endif
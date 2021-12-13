#ifndef TET_VOL_BARRIER_H
#define TET_VOL_BARRIER_H

#include <iostream>

#include "../VecMatDef.h"

void computeVolumeBarrier4Points(double stiffness, const Eigen::Matrix<double,24,1> & cell_vertices, double& energy);
void computeVolumeBarrier4PointsGradient(double stiffness, const Eigen::Matrix<double,24,1> & cell_vertices, Eigen::Matrix<double, 24, 1>& energygradient);
void computeVolumeBarrier4PointsHessian(double stiffness, const Eigen::Matrix<double,24,1> & cell_vertices, Eigen::Matrix<double, 24, 24>& energyhessian);


void computeVolumeBarrier5Points(double stiffness, const Eigen::Matrix<double,30,1> & cell_vertices, double& energy);
void computeVolumeBarrier5PointsGradient(double stiffness, const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolumeBarrier5PointsHessian(double stiffness, const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeVolumeBarrier6Points(double stiffness, const Eigen::Matrix<double,36,1> & cell_vertices, double& energy);
void computeVolumeBarrier6PointsGradient(double stiffness, const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolumeBarrier6PointsHessian(double stiffness, const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 36>& energyhessian);

void computeTetInversionBarrier(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, double& energy);
void computeTetInversionBarrierGradient(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 12, 1>& energygradient);
void computeTetInversionBarrierHessian(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 12, 12>& energyhessian);


void computeTetInversionBarrierFixedCentroid(double stiffness, double dhat, const Eigen::Matrix<double,3,3> & x, const Eigen::Matrix<double,3,1> & centroid, double& energy);
void computeTetInversionBarrierFixedCentroidGradient(double stiffness, double dhat, const Eigen::Matrix<double,3,3> & x, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 9, 1>& energygradient);
void computeTetInversionBarrierFixedCentroidHessian(double stiffness, double dhat, const Eigen::Matrix<double,3,3> & x, const Eigen::Matrix<double,3,1> & centroid, Eigen::Matrix<double, 9, 9>& energyhessian);

#endif
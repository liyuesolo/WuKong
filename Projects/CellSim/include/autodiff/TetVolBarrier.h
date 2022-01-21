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

void computeVolLogBarrier5Points(double stiffness, double dhat, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, double& energy);
void computeVolLogBarrier5PointsGradient(double stiffness, double dhat, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolLogBarrier5PointsHessian(double stiffness, double dhat, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeVolLogBarrier6Points(double stiffness, double dhat, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, double& energy);
void computeVolLogBarrier6PointsGradient(double stiffness, double dhat, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolLogBarrier6PointsHessian(double stiffness, double dhat, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, Eigen::Matrix<double, 36, 36>& energyhessian);

void computeVolumeBarrier7Points(double stiffness, const Eigen::Matrix<double,42,1> & cell_vertices, double& energy);
void computeVolumeBarrier7PointsGradient(double stiffness, const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 1>& energygradient);
void computeVolumeBarrier7PointsHessian(double stiffness, const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 42>& energyhessian);

void computeVolumeBarrier8Points(double stiffness, const Eigen::Matrix<double,48,1> & cell_vertices, double& energy);
void computeVolumeBarrier8PointsGradient(double stiffness, const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 1>& energygradient);
void computeVolumeBarrier8PointsHessian(double stiffness, const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 48>& energyhessian);

void computeTetInvBarrier2Points(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, double& energy);
void computeTetInvBarrier2PointsGradient(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 6, 1>& energygradient);
void computeTetInvBarrier2PointsHessian(double stiffness, double dhat, const Eigen::Matrix<double,3,4> & x, Eigen::Matrix<double, 6, 6>& energyhessian);


void computeVolQubicUnilateralPenalty5Points(double stiffness, double target, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, double& energy);
void computeVolQubicUnilateralPenalty5PointsGradient(double stiffness, double target, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolQubicUnilateralPenalty5PointsHessian(double stiffness, double target, const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,30,1> & mask, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeVolQubicUnilateralPenalty6Points(double stiffness, double target, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, double& energy);
void computeVolQubicUnilateralPenalty6PointsGradient(double stiffness, double target, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolQubicUnilateralPenalty6PointsHessian(double stiffness, double target, const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,36,1> & mask, Eigen::Matrix<double, 36, 36>& energyhessian);
#endif
#ifndef AREA_ENERGY_H
#define AREA_ENERGY_H
#include <iostream>

#include "../VecMatDef.h"

void computeQuadFaceAreaSquaredSum(double stiffness, const Matrix<double,12,1> & face_vertices, double& energy);
void computeQuadFaceAreaSquaredSumGradient(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 1>& energygradient);
void computeQuadFaceAreaSquaredSumHessian(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 12>& energyhessian);

void computePentFaceAreaSquaredSum(double stiffness, const Matrix<double,15,1> & face_vertices, double& energy);
void computePentFaceAreaSquaredSumGradient(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 1>& energygradient);
void computePentFaceAreaSquaredSumHessian(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 15>& energyhessian);

void computeHexFaceAreaSquaredSum(double stiffness, const Matrix<double,18,1> & face_vertices, double& energy);
void computeHexFaceAreaSquaredSumGradient(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 1>& energygradient);
void computeHexFaceAreaSquaredSumHessian(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 18>& energyhessian);

void computeQuadFaceAreaEnergy(double stiffness, const Matrix<double,12,1> & face_vertices, double& energy);
void computeQuadFaceAreaEnergyGradient(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 1>& energygradient);
void computeQuadFaceAreaEnergyHessian(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 12>& energyhessian);


void computePentFaceAreaEnergy(double stiffness, const Matrix<double,15,1> & face_vertices, double& energy);
void computePentFaceAreaEnergyGradient(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 1>& energygradient);
void computePentFaceAreaEnergyHessian(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 15>& energyhessian);

void computeHexFaceAreaEnergy(double stiffness, const Matrix<double,18,1> & face_vertices, double& energy);
void computeHexFaceAreaEnergyGradient(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 1>& energygradient);
void computeHexFaceAreaEnergyHessian(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 18>& energyhessian);

double computeAreaFourPoints(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, const Matrix<double,3,1> & r2, const Matrix<double,3,1> & r3);
void computeAreaFourPointsGradient(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, const Matrix<double,3,1> & r2, const Matrix<double,3,1> & r3, Matrix<double, 12, 1>& energygradient);
void computeAreaFourPointsHessian(const Matrix<double,3,1> & r0, const Matrix<double,3,1> & r1, const Matrix<double,3,1> & r2, const Matrix<double,3,1> & r3, Matrix<double, 12, 12>& energyhessian);

void computeArea4Points(double stiffness, const Matrix<double,12,1> & face_vertices, double& energy);
void computeArea4PointsGradient(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 1>& energygradient);
void computeArea4PointsHessian(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 12>& energyhessian);

void computeArea5Points(double stiffness, const Matrix<double,15,1> & face_vertices, double& energy);
void computeArea5PointsGradient(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 1>& energygradient);
void computeArea5PointsHessian(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 15>& energyhessian);

void computeArea6Points(double stiffness, const Matrix<double,18,1> & face_vertices, double& energy);
void computeArea6PointsGradient(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 1>& energygradient);
void computeArea6PointsHessian(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 18>& energyhessian);

void computeArea4PointsSquared(double stiffness, const Matrix<double,12,1> & face_vertices, double& energy);
void computeArea4PointsSquaredGradient(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 1>& energygradient);
void computeArea4PointsSquaredHessian(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 12>& energyhessian);

void computeArea5PointsSquared(double stiffness, const Matrix<double,15,1> & face_vertices, double& energy);
void computeArea5PointsSquaredGradient(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 1>& energygradient);
void computeArea5PointsSquaredHessian(double stiffness, const Matrix<double,15,1> & face_vertices, Matrix<double, 15, 15>& energyhessian);

void computeArea6PointsSquared(double stiffness, const Matrix<double,18,1> & face_vertices, double& energy);
void computeArea6PointsSquaredGradient(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 1>& energygradient);
void computeArea6PointsSquaredHessian(double stiffness, const Matrix<double,18,1> & face_vertices, Matrix<double, 18, 18>& energyhessian);

void computeArea4PointsSumSquared(double stiffness, const Matrix<double,12,1> & face_vertices, double& energy);
void computeArea4PointsSumSquaredGradient(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 1>& energygradient);
void computeArea4PointsSumSquaredHessian(double stiffness, const Matrix<double,12,1> & face_vertices, Matrix<double, 12, 12>& energyhessian);

// centroid squared sum
void computeArea4PointsSquaredSum(double stiffness, const Eigen::Matrix<double,12,1> & face_vertices, double& energy);
void computeArea4PointsSquaredSumGradient(double stiffness, const Eigen::Matrix<double,12,1> & face_vertices, Eigen::Matrix<double, 12, 1>& energygradient);
void computeArea4PointsSquaredSumHessian(double stiffness, const Eigen::Matrix<double,12,1> & face_vertices, Eigen::Matrix<double, 12, 12>& energyhessian);

void computeArea5PointsSquaredSum(double stiffness, const Eigen::Matrix<double,15,1> & face_vertices, double& energy);
void computeArea5PointsSquaredSumGradient(double stiffness, const Eigen::Matrix<double,15,1> & face_vertices, Eigen::Matrix<double, 15, 1>& energygradient);
void computeArea5PointsSquaredSumHessian(double stiffness, const Eigen::Matrix<double,15,1> & face_vertices, Eigen::Matrix<double, 15, 15>& energyhessian);

void computeArea6PointsSquaredSum(double stiffness, const Eigen::Matrix<double,18,1> & face_vertices, double& energy);
void computeArea6PointsSquaredSumGradient(double stiffness, const Eigen::Matrix<double,18,1> & face_vertices, Eigen::Matrix<double, 18, 1>& energygradient);
void computeArea6PointsSquaredSumHessian(double stiffness, const Eigen::Matrix<double,18,1> & face_vertices, Eigen::Matrix<double, 18, 18>& energyhessian);

void computeArea7PointsSquaredSum(double stiffness, const Eigen::Matrix<double,21,1> & face_vertices, double& energy);
void computeArea7PointsSquaredSumGradient(double stiffness, const Eigen::Matrix<double,21,1> & face_vertices, Eigen::Matrix<double, 21, 1>& energygradient);
void computeArea7PointsSquaredSumHessian(double stiffness, const Eigen::Matrix<double,21,1> & face_vertices, Eigen::Matrix<double, 21, 21>& energyhessian);

void computeArea8PointsSquaredSum(double stiffness, const Eigen::Matrix<double,24,1> & face_vertices, double& energy);
void computeArea8PointsSquaredSumGradient(double stiffness, const Eigen::Matrix<double,24,1> & face_vertices, Eigen::Matrix<double, 24, 1>& energygradient);
void computeArea8PointsSquaredSumHessian(double stiffness, const Eigen::Matrix<double,24,1> & face_vertices, Eigen::Matrix<double, 24, 24>& energyhessian);

#endif
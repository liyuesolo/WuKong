#ifndef CELL_ENERGY_H
#define CELL_ENERGY_H

#include <iostream>

#include "../VecMatDef.h"

void computeVolume4Points(const Matrix<double,24,1> & cell_vertices, double& energy);
void computeVolume4PointsGradient(const Matrix<double,24,1> & cell_vertices, Matrix<double, 24, 1>& energygradient);

void computeVolume4PointsHessian(const Matrix<double,24,1> & cell_vertices, Matrix<double, 24, 24>& energyhessian);
void computeVolume5Points(const Eigen::Matrix<double,30,1> & cell_vertices, double& energy);
void computeVolume5PointsGradient(const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolume5PointsHessian(const Eigen::Matrix<double,30,1> & cell_vertices, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeVolume6Points(const Eigen::Matrix<double,36,1> & cell_vertices, double& energy);
void computeVolume6PointsGradient(const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolume6PointsHessian(const Eigen::Matrix<double,36,1> & cell_vertices, Eigen::Matrix<double, 36, 36>& energyhessian);

void computeHexBasePrismVolume(const Eigen::Matrix<double,36,1> & prism_vertices, double& energy);
void computeHexBasePrismVolumeGradient(const Eigen::Matrix<double,36,1> & prism_vertices, Eigen::Matrix<double, 36, 1>& energygradient);
void computeHexBasePrismVolumeHessian(const Eigen::Matrix<double,36,1> & prism_vertices, Eigen::Matrix<double, 36, 36>& energyhessian);
void computePentaBasePrismVolume(const Eigen::Matrix<double,30,1> & prism_vertices, double& energy);
void computePentaBasePrismVolumeGradient(const Eigen::Matrix<double,30,1> & prism_vertices, Eigen::Matrix<double, 30, 1>& energygradient);
void computePentaBasePrismVolumeHessian(const Eigen::Matrix<double,30,1> & prism_vertices, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeQuadBasePrismVolume(const Eigen::Matrix<double,24,1> & prism_vertices, double& energy);
void computeQuadBasePrismVolumeGradient(const Eigen::Matrix<double,24,1> & prism_vertices, Eigen::Matrix<double, 24, 1>& energygradient);
void computeQuadBasePrismVolumeHessian(const Eigen::Matrix<double,24,1> & prism_vertices, Eigen::Matrix<double, 24, 24>& energyhessian);


void computePentaBasePrismVolumePenalty(double stiffness, const Eigen::Matrix<double,30,1> & prism_vertices, const Eigen::Matrix<double,9,1> & init_tet_vol, double& energy);
void computePentaBasePrismVolumePenaltyGradient(double stiffness, const Eigen::Matrix<double,30,1> & prism_vertices, const Eigen::Matrix<double,9,1> & init_tet_vol, Eigen::Matrix<double, 30, 1>& energygradient);
void computePentaBasePrismVolumePenaltyHessian(double stiffness, const Eigen::Matrix<double,30,1> & prism_vertices, const Eigen::Matrix<double,9,1> & init_tet_vol, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeHexBasePrismVolumePenalty(double stiffness, const Eigen::Matrix<double,36,1> & prism_vertices, const Eigen::Matrix<double,12,1> & init_tet_vol, double& energy);
void computeHexBasePrismVolumePenaltyGradient(double stiffness, const Eigen::Matrix<double,36,1> & prism_vertices, const Eigen::Matrix<double,12,1> & init_tet_vol, Eigen::Matrix<double, 36, 1>& energygradient);
void computeHexBasePrismVolumePenaltyHessian(double stiffness, const Eigen::Matrix<double,36,1> & prism_vertices, const Eigen::Matrix<double,12,1> & init_tet_vol, Eigen::Matrix<double, 36, 36>& energyhessian);

void computeVolume5PointsFixedCentroid(const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,24,1> & centroids, double& energy);
void computeVolume5PointsFixedCentroidGradient(const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,24,1> & centroids, Eigen::Matrix<double, 30, 1>& energygradient);
void computeVolume5PointsFixedCentroidHessian(const Eigen::Matrix<double,30,1> & cell_vertices, const Eigen::Matrix<double,24,1> & centroids, Eigen::Matrix<double, 30, 30>& energyhessian);

void computeVolume6PointsFixedCentroid(const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,27,1> & centroids, double& energy);
void computeVolume6PointsFixedCentroidGradient(const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,27,1> & centroids, Eigen::Matrix<double, 36, 1>& energygradient);
void computeVolume6PointsFixedCentroidHessian(const Eigen::Matrix<double,36,1> & cell_vertices, const Eigen::Matrix<double,27,1> & centroids, Eigen::Matrix<double, 36, 36>& energyhessian);


void computeVolume7Points(const Eigen::Matrix<double,42,1> & cell_vertices, double& energy);
void computeVolume7PointsGradient(const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 1>& energygradient);
void computeVolume7PointsHessian(const Eigen::Matrix<double,42,1> & cell_vertices, Eigen::Matrix<double, 42, 42>& energyhessian);

void computeVolume8Points(const Eigen::Matrix<double,48,1> & cell_vertices, double& energy);
void computeVolume8PointsGradient(const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 1>& energygradient);
void computeVolume8PointsHessian(const Eigen::Matrix<double,48,1> & cell_vertices, Eigen::Matrix<double, 48, 48>& energyhessian);
#endif
#ifndef CODEGEN_H
#define CODEGEN_H

#include <Eigen/Sparse>
#include "VecMatDef.h"

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using VectorXi = Vector<int, Eigen::Dynamic>;

VectorXT evaluate_x_voronoi(const VectorXT &c, const VectorXi &tri);

Eigen::SparseMatrix<double> evaluate_dxdc_voronoi(const VectorXT &c, const VectorXi &tri);

VectorXT evaluate_x_sectional(const VectorXT &c, const VectorXi &tri);

Eigen::SparseMatrix<double> evaluate_dxdc_sectional(const VectorXT &c, const VectorXi &tri);

VectorXT evaluate_A(const VectorXT &c, const VectorXT &x, const VectorXi &e);

Eigen::SparseMatrix<double> evaluate_dAdx(const VectorXT &c, const VectorXT &x, const VectorXi &e);

double evaluate_L(const VectorXT &x, const VectorXi &e);

Eigen::SparseMatrix<double> evaluate_dLdx(const VectorXT &x, const VectorXi &e);

double evaluate_O_voronoi(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

Eigen::SparseMatrix<double>
evaluate_dOdc_voronoi(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

Eigen::SparseMatrix<double>
evaluate_d2Odc2_voronoi(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

double evaluate_O_sectional(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

Eigen::SparseMatrix<double>
evaluate_dOdc_sectional(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

Eigen::SparseMatrix<double>
evaluate_d2Odc2_sectional(const VectorXT &c, const VectorXi &tri, const VectorXi &e, const VectorXT &p);

#endif

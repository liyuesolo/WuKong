#ifndef CODEGEN_H
#define CODEGEN_H

#include <Eigen/Sparse>
#include "VecMatDef.h"

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using VectorXi = Vector<int, Eigen::Dynamic>;

VectorXT evaluate_x(const VectorXT &c, const VectorXi &tri);

Eigen::SparseMatrix<double> evaluate_dxdc(const VectorXT &c, const VectorXi &tri);

VectorXT evaluate_A(const VectorXT &c, const VectorXT &x, const VectorXi &e);

Eigen::SparseMatrix<double> evaluate_dAdx(const VectorXT &c, const VectorXT &x, const VectorXi &e);

#endif

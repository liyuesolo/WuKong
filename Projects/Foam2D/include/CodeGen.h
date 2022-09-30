#ifndef CODEGEN_H
#define CODEGEN_H

#include <Eigen/Sparse>
#include "VecMatDef.h"

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using VectorXi = Vector<int, Eigen::Dynamic>;

void add_O_voronoi_cell(const VectorXT &c, const VectorXT &p, double &out);

void add_dOdc_voronoi_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, VectorXT &out);

void
add_d2Odc2_voronoi_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, Eigen::SparseMatrix<double> &out);

void add_O_sectional_cell(const VectorXT &c, const VectorXT &p, double &out);

void add_dOdc_sectional_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, VectorXT &out);

void
add_d2Odc2_sectional_cell(const VectorXT &c, const VectorXT &p, const VectorXi &map, Eigen::SparseMatrix<double> &out);

#endif

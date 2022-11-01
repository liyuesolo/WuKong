#ifndef CODEGEN_H
#define CODEGEN_H

#include <Eigen/Sparse>
#include "VecMatDef.h"

#include "Tessellation/Tessellation.h"

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using VectorXi = Vector<int, Eigen::Dynamic>;

struct CasadiFunctions;

CasadiFunctions getCasadiFunctions(Tessellation *tessellation, int num_neighbors);

void add_O_cell(Tessellation *tessellation, const VectorXT &c, const VectorXT &p, double &out);

void add_dOdc_cell(Tessellation *tessellation, const VectorXT &c, const VectorXT &p, const VectorXi &map,
                   VectorXT &out);

void
add_d2Odc2_cell(Tessellation *tessellation, const VectorXT &c, const VectorXT &p, const VectorXi &map,
                Eigen::SparseMatrix<double> &out);

#endif

#ifndef CODEGEN_H
#define CODEGEN_H

#include <Eigen/Sparse>
#include "VecMatDef.h"

#include "Tessellation/Tessellation.h"

using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
using VectorXi = Vector<int, Eigen::Dynamic>;

struct CasadiFunctions;

CasadiFunctions getCasadiFunctions(Tessellation *tessellation, int num_neighbors);

void add_E_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
                double &out);

void
add_dEdc_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
              const VectorXi &map,
              VectorXT &out);

void
add_d2Edc2_cell(Tessellation *tessellation, const VectorXT &p, const VectorXT &n, const VectorXT &c, const VectorXT &b,
                const VectorXi &map,
                MatrixXT &out);

#endif

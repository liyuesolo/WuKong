#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H


#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

#include "VecMatDef.h"

namespace LinearSolver
{

    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Entry = Eigen::Triplet<T>;


    bool WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du, bool add_to_diagonal = true, 
         bool check_search_dir = true, bool check_residual = true);
    bool linearSolve(StiffnessMatrix& A,
         const VectorXT& b, VectorXT& x, bool add_to_diagonal = true, 
         bool check_search_dir = true, bool check_residual = true);

     bool linearSolveEigen(StiffnessMatrix& A,
         const VectorXT& b, VectorXT& x, bool add_to_diagonal = true, 
         bool check_search_dir = true, bool check_residual = true);
};


#endif
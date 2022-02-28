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

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
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

    bool solveLUEigen(StiffnessMatrix& A, const VectorXT& b, VectorXT& x);

    template <class Solver>
    bool solve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du, int reg_start, int reg_offset)
    {
        // K = K.selfadjointView<Eigen::Lower>();
        Solver solver;
        T alpha = 10e-6;
        
        solver.analyzePattern(K);
        

        int i = 0;
        for (; i < 50; i++)
        {
            
            solver.factorize(K);
            if (solver.info() != Eigen::Success)
            {
                std::cout << "decomposition failed" << std::endl;
                tbb::parallel_for(reg_start, reg_start + reg_offset, [&](int row)
                {
                    K.coeffRef(row, row) += alpha;
                });  
                // K.diagonal().array() += alpha; 
                alpha *= 10;
                continue;
            }
            
            du = solver.solve(residual);

            T dot_dx_g = du.normalized().dot(residual.normalized());
            
            bool search_dir_correct_sign = dot_dx_g > 1e-6;
            bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
            
            if (search_dir_correct_sign && solve_success)
            {
                return true;
            }
            else
            {
                // tbb::parallel_for(reg_start, reg_start + reg_offset, [&](int row)
                // {
                //     K.coeffRef(row, row) += alpha;
                // }); 
                K.diagonal().array() += alpha;        
                alpha *= 10;
            }
        }
        return false;
    }
};


#endif
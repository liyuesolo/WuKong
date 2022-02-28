#include "../include/LinearSolver.h"
#include <Eigen/PardisoSupport>

bool LinearSolver::linearSolveEigen(StiffnessMatrix& A,
         const VectorXT& b, VectorXT& x, bool add_to_diagonal,
         bool check_search_dir, bool check_residual)
{
    Eigen::SparseLU<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    T alpha = 10e-6;
    solver.analyzePattern(A);
    int i = 0;
    for (; i < 50; i++)
    {
        solver.factorize(A);
        if (solver.info() == Eigen::NumericalIssue)
        {
            std::cout << "indefinite" << std::endl;
            tbb::parallel_for(0, (int)A.rows(), [&](int row)    
            {
                A.coeffRef(row, row) += alpha;
            });  
            std::cout << "add to diagonal" << std::endl;
            alpha *= 10;
            continue;
        }
        x = solver.solve(b);
        if (!check_search_dir && !check_residual)
            return true;

        T dot_dx_g = x.normalized().dot(b.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (A*x - b).norm() < 1e-6 && solver.info() == Eigen::Success;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            return true;
        }
        else
        {
            tbb::parallel_for(0, (int)A.rows(), [&](int row)    
            {
                A.coeffRef(row, row) += alpha;
            });        
            alpha *= 10;
        }
    }
    return false;
}

bool LinearSolver::linearSolve(StiffnessMatrix& A,
         const VectorXT& b, VectorXT& x, bool add_to_diagonal,
         bool check_search_dir, bool check_residual)
{
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;
    T alpha = 10e-6;
    solver.analyzePattern(A);
    int i = 0;
    for (; i < 50; i++)
    {
        solver.factorize(A);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            tbb::parallel_for(0, (int)A.rows(), [&](int row)    
            {
                A.coeffRef(row, row) += alpha;
            });  
            alpha *= 10;
            continue;
        }
        x = solver.solve(b);
        if (!check_search_dir && !check_residual)
            return true;

        T dot_dx_g = x.normalized().dot(b.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (A*x - b).norm() < 1e-6 && solver.info() == Eigen::Success;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            return true;
        }
        else
        {
            tbb::parallel_for(0, (int)A.rows(), [&](int row)    
            {
                A.coeffRef(row, row) += alpha;
            });        
            alpha *= 10;
        }
    }
    return false;
}
bool LinearSolver::solveLUEigen(StiffnessMatrix& A, const VectorXT& b, VectorXT& x)
{
    Eigen::SparseLU<StiffnessMatrix> solver;
    solver.analyzePattern(A);
    solver.factorize(A);
    x = solver.solve(b);
}

bool LinearSolver::WoodburySolve(StiffnessMatrix& K, const MatrixXT& UV,
         VectorXT& residual, VectorXT& du, bool add_to_diagonal, 
         bool check_search_dir, bool check_residual)
{
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, int>> solver;

    T alpha = 10e-6;
    solver.analyzePattern(K);
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        // std::cout << "-----factorization takes " << t.elapsed_sec() << "s----" << std::endl;
        if (solver.info() == Eigen::NumericalIssue)
        {
            // K = H + alpha * I;        
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            }); 
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        // sherman morrison
        if (UV.cols() == 1)
        {
            VectorXT v = UV.col(0);
            VectorXT A_inv_g = solver.solve(residual);
            VectorXT A_inv_u = solver.solve(v);

            T dem = 1.0 + v.dot(A_inv_u);

            du = A_inv_g - (A_inv_g.dot(v)) * A_inv_u / dem;
        }
        // UV is actually only U, since UV is the same in the case
        // C is assume to be Identity
        else // Woodbury https://en.wikipedia.org/wiki/Woodbury_matrix_identity
        {
            VectorXT A_inv_g = solver.solve(residual);

            MatrixXT A_inv_U(UV.rows(), UV.cols());
            for (int col = 0; col < UV.cols(); col++)
                A_inv_U.col(col) = solver.solve(UV.col(col));
            
            MatrixXT C(UV.cols(), UV.cols());
            C.setIdentity();
            C += UV.transpose() * A_inv_U;
            du = A_inv_g - A_inv_U * C.inverse() * UV.transpose() * A_inv_g;
        }
        
        if (!check_search_dir && !check_residual)
            return true;

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        bool solve_success = (K * du + UV * UV.transpose()*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            return true;
        }
        else
        {
            // K = H + alpha * I;       
            tbb::parallel_for(0, (int)K.rows(), [&](int row)    
            {
                K.coeffRef(row, row) += alpha;
            });  
            alpha *= 10;
        }
    }
    return false;
}
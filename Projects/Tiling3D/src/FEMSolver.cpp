#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

#include <Eigen/PardisoSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

T FEMSolver::computeTotalEnergy(const VectorXT& u)
{
    T total_energy = 0.0;

    VectorXT projected = u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    VectorXT energies_neoHookean = VectorXT::Zero(num_ele);

    iterateTetsParallel([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        T ei;
        computeLinearTet3DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
    });

    total_energy += energies_neoHookean.sum();

    return total_energy;
}


T FEMSolver::computeResidual(const VectorXT& u, VectorXT& residual)
{
    VectorXT projected = u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }

    deformed = undeformed + projected;
    
    iterateTetsSerial([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        Vector<T, 12> dedx;
        computeLinearTet3DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
        addForceEntry<12>(residual, indices, -dedx);
    });

    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
        
    return residual.norm();
}

void FEMSolver::buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K)
{
    VectorXT projected = u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;
    
    std::vector<Entry> entries;
    
    iterateTetsSerial([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        Matrix<T, 12, 12> hessian;
        computeLinearTet3DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
        addHessianEntry<12>(entries, indices, hessian);
    });

    K.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();

}

void FEMSolver::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

bool FEMSolver::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

    Eigen::PardisoLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    T mu = 10e-6;
    solver.analyzePattern(K);
    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            // std::cout << "indefinite" << std::endl;
            K = H + mu * I;        
            mu *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-3;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            break;
        else
        {
            K = H + mu * I;        
            mu *= 10;
        }
        if (i == 49)
            return false;
    }
    return true;
}

T FEMSolver::lineSearchNewton(VectorXT& u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(u, K);
    
    // Timer ti(true);
    bool success = linearSolve(K, residual, du);
    // std::cout << ti.elapsed_sec() << std::endl;
    if (!success)
        return 1e16;
    T norm = du.norm();
    // std::cout << du.norm() << std::endl;
    T alpha = 1;
    T E0 = computeTotalEnergy(u);
    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > 15)
        {
            // if (cnt > 15)
            //     std::cout << "cnt > 15" << std::endl;
            u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    return norm;
}

bool FEMSolver::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    while (true)
    {
        VectorXT residual(deformed.rows());
        residual.setZero();

        residual_norm = computeResidual(u, residual);
        
        // if (verbose)
            std::cout << "residual_norm " << residual_norm << std::endl;
        
        if (residual_norm < newton_tol)
            break;
        
        dq_norm = lineSearchNewton(u, residual);

        if(cnt == max_newton_iter || dq_norm > 1e10)
            break;
        cnt++;
    }

    iterateDirichletDoF([&](int offset, T target)
    {
        u[offset] = target;
    });
    deformed = undeformed + u;

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|ddu|: " << dq_norm  << std::endl;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}
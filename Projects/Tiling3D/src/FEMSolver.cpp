#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"

#include <Eigen/PardisoSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

T FEMSolver::computeInteralEnergy(const VectorXT& _u)
{
    VectorXT projected = _u;
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

    return energies_neoHookean.sum();
}

T FEMSolver::computeTotalEnergy(const VectorXT& _u)
{
    T total_energy = 0.0;

    VectorXT projected = _u;
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

    total_energy -= _u.dot(f);

    if (use_ipc)
    {
        T contact_energy = 0.0;
        addIPCEnergy(contact_energy);
        total_energy += contact_energy;
    }

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

    residual = f;
    
    iterateTetsSerial([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        Vector<T, 12> dedx;
        computeLinearTet3DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);
        addForceEntry<12>(residual, indices, -dedx);
    });

    if (use_ipc)
    {
        addIPCForceEntries(residual);
    }

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
    
    if (use_ipc)
    {
        addIPCHessianEntries(entries);
    }

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

T FEMSolver::lineSearchNewton(VectorXT& _u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(_u, K);
    
    // Timer ti(true);
    bool success = linearSolve(K, residual, du);
    // std::cout << ti.elapsed_sec() << std::endl;
    if (!success)
        return 1e16;
    T norm = du.norm();
    // std::cout << du.norm() << std::endl;
    T alpha = computeInversionFreeStepsize(_u, du);
    std::cout << "** step size **" << std::endl;
    std::cout << "after tet inv step size: " << alpha << std::endl;
    if (use_ipc)
    {
        T ipc_step_size = computeCollisionFreeStepsize(_u, du);
        alpha = std::min(alpha, ipc_step_size);
        std::cout << "after ipc step size: " << alpha << std::endl;
    }
    std::cout << "**       **" << std::endl;
    T E0 = computeTotalEnergy(_u);
    int cnt = 0;
    while (true)
    {
        VectorXT u_ls = _u + alpha * du;
        T E1 = computeTotalEnergy(u_ls);
        if (E1 - E0 < 0 || cnt > 15)
        {
            // if (cnt > 15)
            //     std::cout << "cnt > 15" << std::endl;
            _u = u_ls;
            break;
        }
        alpha *= 0.5;
        cnt += 1;
    }
    return norm;
}

T FEMSolver::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
           
    VectorXT step_sizes = VectorXT::Zero(num_ele);

    iterateTetsParallel([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        TM dXdb = x_undeformed.transpose() * dNdb;
        TM dxdb = x_deformed.transpose() * dNdb;
        TM A = dxdb * dXdb.inverse();
        T a, b, c, d;
        a = A.determinant();
        b = A(0, 0) * A(1, 1) - A(0, 1) * A(1, 0) + A(0, 0) * A(2, 2) - A(0, 2) * A(2, 0) + A(1, 1) * A(2, 2) - A(1, 2) * A(2, 1);
        c = A.diagonal().sum();
        d = 0.8;

        T t = getSmallestPositiveRealCubicRoot(a, b, c, d);
        if (t < 0 || t > 1) t = 1;
            step_sizes(tet_idx) = t;
    });
    return step_sizes.minCoeff();
}

void FEMSolver::incrementalLoading()
{
    std::unordered_map<int, T> dirichlet_bc_target = dirichlet_data;

    int n_step = 50;
    for (int step = 1; step <= n_step; step++)
    {
        for (auto& data : dirichlet_data)
            data.second = T(1) / T(n_step) * dirichlet_bc_target[data.first];
        staticSolve();
        // saveToOBJ("iter_" + std::to_string(step) + ".obj");
        undeformed = deformed;
        u.setZero();
    }
    
}

bool FEMSolver::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    iterateDirichletDoF([&](int offset, T target)
    {
        f[offset] = 0;
    });

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
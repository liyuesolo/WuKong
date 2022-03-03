#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"
#include "../include/Timer.h"
#include <Eigen/PardisoSupport>

#include "../../Solver/CHOLMODSolver.hpp"

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

#include <fstream>
#include <iomanip>

template <int dim>
T FEMSolver<dim>::computeInteralEnergy(const VectorXT& _u)
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

    iterateElementParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
    {
        T ei;
        computeLinearTet3DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
    });

    return energies_neoHookean.sum();
}

template <int dim>
T FEMSolver<dim>::computeTotalEnergy(const VectorXT& _u)
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

    T e_NH = 0.0;
    addElastsicPotential(e_NH);
    total_energy += e_NH;

    if (use_penalty)
    {
        T penalty_energy = 0.0;
        addBCPenaltyEnergy(penalty_energy);
        total_energy += penalty_energy;
    }

    total_energy -= _u.dot(f);

    if (use_ipc)
    {
        T contact_energy = 0.0;
        addIPCEnergy(contact_energy);
        total_energy += contact_energy;
    }

    return total_energy;
}

template <int dim>
T FEMSolver<dim>::computeResidual(const VectorXT& _u, VectorXT& residual)
{
    
    VectorXT projected = _u;

    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    if (use_ipc)
        updateIPCVertices(_u);

    deformed = undeformed + projected;

    residual = f;
    
    VectorXT residual_backup = residual;

    addElasticForceEntries(residual);

    std::cout << "elastic force " << (residual - residual_backup).norm() << std::endl;
    residual_backup = residual;

    if (use_penalty)
        addBCPenaltyForceEntries(residual);

    if (use_ipc)
    {
        addIPCForceEntries(residual);
        std::cout << "contact + penalty force " << (residual - residual_backup).norm() << std::endl;
        residual_backup = residual;
    }

    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
        
    return residual.norm();
}

template <int dim>
void FEMSolver<dim>::reset()
{
    deformed = undeformed;
    u.setZero();
    
    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    num_ipc_vtx = ipc_vertices.rows();
}

template <int dim>
void FEMSolver<dim>::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
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
    
    std::vector<Entry> entries;


    addElasticHessianEntries(entries);

    if (use_penalty)
        addBCPenaltyHessianEntries(entries);
    
    if (use_ipc)
    {
        addIPCHessianEntries(entries, project_block_PD);
    }

    K.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();

}

template <int dim>
void FEMSolver<dim>::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}

template <int dim>
bool FEMSolver<dim>::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{

    Timer t(true);
    // StiffnessMatrix I(K.rows(), K.cols());
    // I.setIdentity();
    T alpha = 10e-6;
    // StiffnessMatrix H = K;
    std::unordered_map<int, int> diagonal_entry_location;
    
    // Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    Eigen::PardisoLLT<Eigen::SparseMatrix<T, Eigen::RowMajor, int>> solver;
    
    solver.analyzePattern(K);
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    
    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            indefinite_count_reg_cnt++;
            tbb::parallel_for(0, (int)K.rows(), [&](int row)
            {
                K.coeffRef(row, row) += alpha;
            });
            
            // K = H + alpha * I;        
            alpha *= 10;
            continue;
        }
        du = solver.solve(residual);

        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-5;
        if (!search_dir_correct_sign)
            invalid_search_dir_cnt++;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;
        if (!solve_success)
            invalid_residual_cnt++;
        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            std::cout << "\tnnz: " << K.nonZeros() << std::endl;
            std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
            std::cout << "\t# regularization step " << i 
                << " indefinite " << indefinite_count_reg_cnt 
                << " invalid search dir " << invalid_search_dir_cnt
                << " invalid solve " << invalid_residual_cnt << std::endl;
            std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
            std::cout << "\t======================== " << std::endl;
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

template <int dim>
T FEMSolver<dim>::lineSearchNewton(VectorXT& _u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    Timer ti(true);
    buildSystemMatrix(_u, K);
    std::cout << "build system takes " <<  ti.elapsed_sec() << std::endl;
    bool success = linearSolve(K, residual, du);
    
    if (!success)
        return 1e16;
    T norm = du.norm();
    std::cout << du.norm() << std::endl;
    
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
    std::cout << "#ls " << cnt << " alpha = " << alpha << std::endl;
    return norm;
}

template <int dim>
T FEMSolver<dim>::computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du)
{
    Matrix<T, 4, 3> dNdb;
        dNdb << -1.0, -1.0, -1.0, 
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0;
           
    VectorXT step_sizes = VectorXT::Zero(num_ele);

    iterateElementParallel([&](const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, const EleIdx& indices, int tet_idx)
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

template <int dim>
bool FEMSolver<dim>::staticSolveStep(int step)
{
    if (step == 0)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            f[offset] = 0;
        });
        u.setRandom();
        u *= 1.0 / u.norm();
        u *= 0.001;
    }

    VectorXT residual(deformed.rows());
    residual.setZero();

    T residual_norm = computeResidual(u, residual);
    // saveIPCMesh("/home/yueli/Documents/ETH/WuKong/output/ThickShell/IPC_mesh_iter_" + std::to_string(step) + ".obj");
    // saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/iter_" + std::to_string(step) + ".obj");
    std::cout << "iter " << step << "/" << max_newton_iter << ": residual_norm " << residual_norm << " tol: " << newton_tol << std::endl;

    if (residual_norm < newton_tol)
        return true;
    

    T dq_norm = lineSearchNewton(u, residual);
    // saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/iter_" + std::to_string(step) + ".obj");
    saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/structure_iter_" + std::to_string(step) + ".obj");
    // iterateDirichletDoF([&](int offset, T target)
    // {
    //     u[offset] = target;
    // });
    // deformed = undeformed + u;

    if(step == max_newton_iter || dq_norm > 1e10)
        return true;
    
    return false;

}

template <int dim>
bool FEMSolver<dim>::staticSolve()
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
        // saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/iter_" + std::to_string(cnt) + ".obj");
        
        if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter 
            << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
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

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " 
        << residual_norm << "|ddu|: " << dq_norm  << std::endl;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}

// template class FEMSolver<2>;
template class FEMSolver<3>;
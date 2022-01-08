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

void FEMSolver::computeLinearModes()
{
    int nmodes = 20;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);

    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);

        //0 cannot cannot be used as a shift
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("fem_eigen_vectors.txt");
            out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
            for (int i = 0; i < eigen_vectors.cols(); i++)
                out << eigen_values[eigen_vectors.cols() - 1 - i] << " ";
            out << std::endl;
            for (int i = 0; i < eigen_vectors.rows(); i++)
            {
                // for (int j = 0; j < eigen_vectors.cols(); j++)
                for (int j = eigen_vectors.cols() - 1; j >-1 ; j--)
                    out << eigen_vectors(i, j) << " ";
                out << std::endl;
            }       
            out << std::endl;
            out.close();
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
    else
    {
        Eigen::MatrixXd A_dense = K;
        Eigen::EigenSolver<Eigen::MatrixXd> eigen_solver;
        eigen_solver.compute(A_dense, /* computeEigenvectors = */ true);
        auto eigen_values = eigen_solver.eigenvalues();
        auto eigen_vectors = eigen_solver.eigenvectors();
        
        std::vector<T> ev_all(A_dense.cols());
        for (int i = 0; i < A_dense.cols(); i++)
        {
            ev_all[i] = eigen_values[i].real();
        }
        
        std::vector<int> indices;
        for (int i = 0; i < A_dense.cols(); i++)
        {
            indices.push_back(i);    
        }
        std::sort(indices.begin(), indices.end(), [&ev_all](int a, int b){ return ev_all[a] < ev_all[b]; } );
        // std::sort(ev_all.begin(), ev_all.end());

        for (int i = 0; i < nmodes; i++)
            std::cout << ev_all[indices[i]] << std::endl;
        

        std::ofstream out("fem_eigen_vectors.txt");
        out << nmodes << " " << A_dense.cols() << std::endl;
        for (int i = 0; i < nmodes; i++)
            out << ev_all[indices[i]] << " ";
        out << std::endl;
        for (int i = 0; i < nmodes; i++)
        {
            out << eigen_vectors.col(indices[i]).real() << std::endl;
        }

        out.close();
    }
}

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
        if (tet_idx < cylinder_tet_start)
            computeLinearTet3DNeoHookeanEnergy(E, nu, x_deformed, x_undeformed, ei);
        else
            computeLinearTet3DNeoHookeanEnergy(E_steel, nu_steel, x_deformed, x_undeformed, ei);
        energies_neoHookean[tet_idx] += ei;
    });

    total_energy += energies_neoHookean.sum();

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


T FEMSolver::computeResidual(const VectorXT& _u, VectorXT& residual)
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

    iterateTetsSerial([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        // std::cout << indices.transpose() << std::endl;
        Vector<T, 12> dedx;
        if (tet_idx < cylinder_tet_start)
            computeLinearTet3DNeoHookeanEnergyGradient(E, nu, x_deformed, x_undeformed, dedx);  
        else
            computeLinearTet3DNeoHookeanEnergyGradient(E_steel, nu_steel, x_deformed, x_undeformed, dedx);  
        
        addForceEntry<12>(residual, indices, -dedx);
    });

    std::cout << "elastic force " << (residual - residual_backup).norm() << std::endl;
    residual_backup = residual;

    if (use_penalty)
        addBCPenaltyForceEntries(residual);

    if (use_ipc)
    {
        addIPCForceEntries(residual);
        std::cout << "contact force " << (residual - residual_backup).norm() << std::endl;
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

void FEMSolver::reset()
{
    deformed = undeformed;
    u.setZero();
    
    ipc_vertices.resize(num_nodes, 3);
    for (int i = 0; i < num_nodes; i++)
        ipc_vertices.row(i) = undeformed.segment<3>(i * 3);
    num_ipc_vtx = ipc_vertices.rows();
}

void FEMSolver::runBendingHomogenization()
{
    int n_angles = 50;
    T cycle = 1. * M_PI;
    std::vector<T> thetas;
    std::vector<T> stiffness;
    for (T theta = 0; theta <= cycle; theta += cycle/(T)n_angles)
    {
        T theta6 = std::round( theta * 1e4 ) / 1e4;
        thetas.push_back(theta6);
        // std::cout << theta6 << std::endl;

        bending_direction = theta6;
        computeCylindricalBendingBCPenaltyPairs();
        staticSolve();
        T bending_stiffness = computeBendingStiffness();
        
        stiffness.push_back(bending_stiffness);
        reset();
    }
    int n_theta = thetas.size();
    // std::cout << n_theta << std::endl;
    for (int i = 1; i < n_angles; i++)
        thetas.push_back(M_PI + thetas[i]);
    thetas.push_back(M_PI * 2.0);
    for (int i = 1; i < n_angles; i++)
        stiffness.push_back(stiffness[i]);
    stiffness.push_back(stiffness[0]);
    for(T theta : thetas)
        std::cout << std::setprecision(6) << theta << " ";
    std::cout << std::endl;
    for(T k : stiffness)
        std::cout << std::setprecision(6) << k << " ";
    std::cout << std::endl;
}

T FEMSolver::computeBendingStiffness()
{
    deformed = undeformed + u;
    T elastic_potential = computeInteralEnergy(u);
    T macro_volume = (max_corner - min_corner).prod();
    T energy_density = elastic_potential / macro_volume;
    T bending_stiffness = 2.0 * energy_density / (curvature * curvature);
    return bending_stiffness;
}

void FEMSolver::buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K)
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
    
    iterateTetsSerial([&](const TetNodes& x_deformed, 
        const TetNodes& x_undeformed, const TetIdx& indices, int tet_idx)
    {
        Matrix<T, 12, 12> hessian;
        if (tet_idx < cylinder_tet_start)
        {
            computeLinearTet3DNeoHookeanEnergyHessian(E, nu, x_deformed, x_undeformed, hessian);
            // std::cout << "structure hessian PD " << isHessianBlockPD<12>(hessian) << std::endl;
            // std::cout << hessian << std::endl;
            // saveTetOBJ("test_tet.obj", x_deformed);
            // saveTetOBJ("test_tet_rest.obj", x_undeformed);
            // std::getchar();
        }
        else
        {
            computeLinearTet3DNeoHookeanEnergyHessian(E_steel, nu_steel, x_deformed, x_undeformed, hessian);
            // std::cout << "cylinder hessian PD " << isHessianBlockPD<12>(hessian) << std::endl;
        }
        if (project_block_PD)
            projectBlockPD<12>(hessian);
        
        addHessianEntry<12>(entries, indices, hessian);
    });

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
#define USE_CHOLMOD_SOLVER
    Timer t(true);
    // StiffnessMatrix I(K.rows(), K.cols());
    // I.setIdentity();
    T alpha = 10e-6;
    // StiffnessMatrix H = K;
    std::unordered_map<int, int> diagonal_entry_location;
    
#ifdef USE_CHOLMOD_SOLVER
    
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;
    Noether::CHOLMODSolver<typename StiffnessMatrix::StorageIndex> cholmod_solver;
    cholmod_solver.set_pattern(K);
    cholmod_solver.analyze_pattern();

    for (int i = 0; i < 50; i++)
    {
        
        if (!cholmod_solver.factorize())
        {
            // std::cout << "indefinite" << std::endl;
            indefinite_count_reg_cnt++;
            tbb::parallel_for(0, (int)K.rows(), [&](int row)
            {
                K.coeffRef(row, row) += alpha;
            });    
            alpha *= 10;
            // cholmod_solver.A->x = K.valuePtr();
            continue;
        }
        
        cholmod_solver.solve(residual.data(), du.data(), true);
        
        VectorXT linSys_error = VectorXT::Zero(du.rows());
        VectorXT ones = VectorXT::Ones(du.rows());
        cholmod_solver.multiply(du.data(), linSys_error.data());
        linSys_error -= residual;
        bool solve_success = linSys_error.norm() < 1e-6;
        if (!solve_success)
            invalid_residual_cnt++;
        if (solve_success)
        {
            t.stop();
            std::cout << "\t===== Linear Solve ===== " << std::endl;
            
            std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
            std::cout << "\t# regularization step " << i 
                << " indefinite " << indefinite_count_reg_cnt 
                << " invalid solve " << invalid_residual_cnt << std::endl;
            std::cout << "\t======================== " << std::endl;
            return true;
        }
        else
        {
            std::cout << "|Ax-b|: " << linSys_error.norm() << std::endl;
            tbb::parallel_for(0, (int)K.rows(), [&](int row)
            {
                K.coeffRef(row, row) += alpha;
            });
            // cholmod_solver.A->x = K.valuePtr();
            // cholmod_solver.A->x[0] += alpha;
            // std::cout << mat_value[0] << std::endl;
            alpha *= 10;
        }

    }
    du = residual.normalized();
    std::cout << "linear solve failed" << std::endl;
    return true;
#else
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
#endif
}

T FEMSolver::lineSearchNewton(VectorXT& _u, VectorXT& residual)
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

bool FEMSolver::staticSolveStep(int step)
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
    if (three_point_bending_with_cylinder)
        saveThreePointBendingData("/home/yueli/Documents/ETH/WuKong/output/ThickShell/", step);
    else    
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
        if (cnt == 0)
        {
            u.setRandom(); u *= 1.0 / u.norm(); u *= 0.001;
        }
        VectorXT residual(deformed.rows());
        residual.setZero();

        residual_norm = computeResidual(u, residual);
        // saveToOBJ("/home/yueli/Documents/ETH/WuKong/output/ThickShell/iter_" + std::to_string(cnt) + ".obj");
        
        // if (verbose)
            std::cout << "iter " << cnt << "/" << max_newton_iter << ": residual_norm " << residual.norm() << " tol: " << newton_tol << std::endl;
        
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


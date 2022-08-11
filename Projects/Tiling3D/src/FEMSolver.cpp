#include <igl/readOBJ.h>
#include "../include/FEMSolver.h"
#include "../include/autodiff/FEMEnergy.h"
#include "../include/Timer.h"
#include <Eigen/PardisoSupport>
#include <Eigen/CholmodSupport>
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
    

    deformed = undeformed + projected;

    residual = f;
    
    VectorXT residual_backup = residual;

    addElasticForceEntries(residual);

    std::cout << "elastic force " << (residual - residual_backup).norm() << std::endl;
    residual_backup = residual;

    if (use_penalty)
    {
        addBCPenaltyForceEntries(residual);
        std::cout << "penalty force " << (residual - residual_backup).norm() << std::endl;
        residual_backup = residual;
    }

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

void FEMSolver::runForceCurvatureExperiment()
{
    T dkappa = 0.1;
    std::vector<T> curvature_values;
    std::vector<T> force_norms;
    for (T kappa = 1.0; kappa < 4; kappa += dkappa)
    {
        curvature = kappa;
        bending_direction = 45.0 / 180.0 * M_PI;
        computeCylindricalBendingBCPenaltyPairs();
        staticSolve();
        VectorXT elastic_force(num_nodes * dim);
        elastic_force.setZero();
        addElasticForceEntries(elastic_force);
        // std::cout << elastic_force.norm() << std::endl;
        // std::getchar();
        curvature_values.push_back(kappa);
        force_norms.push_back(elastic_force.norm() * 0.1);
    }
    for (T v : curvature_values)
        std::cout << v << " ";
    std::cout << std::endl;
    for (T v : force_norms)
        std::cout << v << " ";
    std::cout << std::endl;
}

void FEMSolver::loadForceDisplacementResults()
{
    
    T dp = 0.01;
    std::vector<T> displacements;
    std::vector<T> force_norms;
    VectorXT u_prev = u;
    std::string data_folder = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling3D/";
    for (T percent = 0.01; percent < 0.38; percent += dp)
    // T percent = 0.25;
    {
        T displacement_sum = 0.0;
        penaltyInPlaneCompression(0, percent);
        for (auto pair : penalty_pairs)
        {
            displacement_sum += std::abs(undeformed[pair.first] - pair.second);
        }
        displacement_sum /= T(penalty_pairs.size());
        // u = u_prev;
        // staticSolve();
        // u_prev = u;
        Eigen::MatrixXd Vi; Eigen::MatrixXi Fi;
        igl::readOBJ(data_folder + std::to_string(percent) + ".obj", Vi, Fi);
        for (int i = 0; i < Vi.rows(); i++)
        {
            deformed.segment<3>(i * 3) = Vi.row(i);
        }
        u = deformed - undeformed;
        updateIPCVertices(u);
        VectorXT elastic_force(num_nodes * dim);
        elastic_force.setZero();
        // addElasticForceEntries(elastic_force);
        addBCPenaltyForceEntries(elastic_force);
        displacements.push_back(displacement_sum);
        force_norms.push_back(elastic_force.norm() * 0.1);

        // saveToOBJ(data_folder + std::to_string(percent) + ".obj");
        // break;
    }
    for (T v : displacements)
        std::cout << v << " ";
    std::cout << std::endl;
    for (T v : force_norms)
        std::cout << v << " ";
    std::cout << std::endl;
}

void FEMSolver::runForceDisplacementExperiment()
{
    T dp = 0.01;
    std::vector<T> displacements;
    std::vector<T> force_norms;
    VectorXT u_prev = u;
    std::string data_folder = "/home/yueli/Documents/ETH/WuKong/build/Projects/Tiling3D/exp2/";
    for (T percent = 0.01; percent < 0.4; percent += dp)
    {
        T displacement_sum = 0.0;
        penaltyInPlaneCompression(0, percent);
        for (auto pair : penalty_pairs)
        {
            displacement_sum += std::abs(undeformed[pair.first] - pair.second);
        }
        displacement_sum /= T(penalty_pairs.size());
        u = u_prev;
        staticSolve();
        u_prev = u;
        VectorXT elastic_force(num_nodes * dim);
        elastic_force.setZero();
        addElasticForceEntries(elastic_force);
        displacements.push_back(displacement_sum);
        force_norms.push_back(elastic_force.norm() * 0.1);

        saveToOBJ(data_folder + std::to_string(percent) + ".obj");
        // break;
    }
    for (T v : displacements)
        std::cout << v << " ";
    std::cout << std::endl;
    for (T v : force_norms)
        std::cout << v << " ";
    std::cout << std::endl;
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
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    T alpha = 1e-6;
    solver.analyzePattern(K);
    // T time_analyze = t.elapsed_sec();
    // std::cout << "\t analyzePattern takes " << time_analyze << "s" << std::endl;
    
    int indefinite_count_reg_cnt = 0, invalid_search_dir_cnt = 0, invalid_residual_cnt = 0;

    for (int i = 0; i < 50; i++)
    {
        solver.factorize(K);
        if (solver.info() == Eigen::NumericalIssue)
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
            indefinite_count_reg_cnt++;
            continue;
        }

        du = solver.solve(residual);
        
        T dot_dx_g = du.normalized().dot(residual.normalized());

        int num_negative_eigen_values = 0;
        int num_zero_eigen_value = 0;

        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        if (!search_dir_correct_sign)
        {   
            invalid_search_dir_cnt++;
        }
        
        bool solve_success = true;
        // solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        
        if (!solve_success)
            invalid_residual_cnt++;
        // std::cout << "PD: " << positive_definte << " direction " 
        //     << search_dir_correct_sign << " solve " << solve_success << std::endl;

        if (positive_definte && search_dir_correct_sign && solve_success)
        {
            t.stop();
            if (verbose)
            {
                std::cout << "\t===== Linear Solve ===== " << std::endl;
                std::cout << "\tnnz: " << K.nonZeros() << std::endl;
                std::cout << "\t takes " << t.elapsed_sec() << "s" << std::endl;
                std::cout << "\t# regularization step " << i 
                    << " indefinite " << indefinite_count_reg_cnt 
                    << " invalid search dir " << invalid_search_dir_cnt
                    << " invalid solve " << invalid_residual_cnt << std::endl;
                std::cout << "\tdot(search, -gradient) " << dot_dx_g << std::endl;
                // std::cout << (K.selfadjointView<Eigen::Lower>() * du + UV * UV.transpose()*du - residual).norm() << std::endl;
                std::cout << "\t======================== " << std::endl;
            }
            return true;
        }
        else
        {
            K.diagonal().array() += alpha;
            alpha *= 10;
        }
    }
    return false;
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
        if (E1 - E0 < 0 || cnt > 10)
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
        // u.setRandom();
        // u *= 1.0 / u.norm();
        // u *= 0.001;
    }

    VectorXT residual(deformed.rows());
    residual.setZero();

    T residual_norm = computeResidual(u, residual);
    if (use_ipc)
        updateIPCVertices(u);
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
        
        VectorXT residual(deformed.rows());
        residual.setZero();

        residual_norm = computeResidual(u, residual);
        if (use_ipc)
            updateIPCVertices(u);
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


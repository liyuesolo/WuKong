
#include <Eigen/CholmodSupport>
#include <Eigen/PardisoSupport>
#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>
#include <fstream>
#include "../include/HexFEMSolver.h"
#include "../include/autodiff/HexFEM.h"

#include "../include/Timer.h"

void HexFEMSolver::checkHessianPD(bool save_txt)
{
    
    int nmodes = 10;
    int n_dof_sim = deformed.rows();
    StiffnessMatrix d2edx2(n_dof_sim, deformed.rows());
    buildSystemMatrix(u, d2edx2);
    
    bool use_Spectra = true;

    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    solver.analyzePattern(d2edx2); 
    // std::cout << "analyzePattern" << std::endl;
    solver.factorize(d2edx2);
    // std::cout << "factorize" << std::endl;
    bool indefinite = false;
    if (solver.info() == Eigen::NumericalIssue)
    {
        std::cout << "!!!indefinite matrix!!!" << std::endl;
        indefinite = true;
        
    }
    else
    {
        // std::cout << "indefinite" << std::endl;
    }
    
    if (use_Spectra)
    {
        
        Spectra::SparseSymShiftSolve<T, Eigen::Lower> op(d2edx2);
        // T shift = indefinite ? -1e2 : -1e-4;
        T shift = -1e-4;
        Spectra::SymEigsShiftSolver<T, 
        Spectra::LARGEST_MAGN, 
        Spectra::SparseSymShiftSolve<T, Eigen::Lower> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();

        int nconv = eigs.compute();

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values.transpose() << std::endl;
            if (save_txt)
            {
                std::ofstream out("eigen_vectors.txt");
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
        }
        else
        {
            std::cout << "Eigen decomposition failed" << std::endl;
        }
    }
}


void HexFEMSolver::computeEigenMode()
{
    int nmodes = 15;

    StiffnessMatrix K(deformed.rows(), deformed.rows());
    run_diff_test = true;
    buildSystemMatrix(u, K);
    // for (auto iter : dirichlet_data)
    // {
    //     K.coeffRef(iter.first, iter.first) = 1e10;
    // }
    std::cout << "build matrix" << std::endl;

    // std::cout << K << std::endl;
    bool use_Spectra = true;

    if (use_Spectra)
    {

        Spectra::SparseSymShiftSolve<T, Eigen::Upper> op(K);
        
        std::cout << "pass K" << std::endl;

        //0 cannot cannot be used as a shift
        T shift = -0.1;
        Spectra::SymEigsShiftSolver<T, 
            Spectra::LARGEST_MAGN, 
            Spectra::SparseSymShiftSolve<T, Eigen::Upper> > 
            eigs(&op, nmodes, 2 * nmodes, shift);

        eigs.init();
        
        std::cout << "init" << std::endl;

        int nconv = eigs.compute();

        std::cout << "compute" << std::endl;

        if (eigs.info() == Spectra::SUCCESSFUL)
        {
            Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().real();
            Eigen::VectorXd eigen_values = eigs.eigenvalues().real();
            std::cout << eigen_values << std::endl;
            std::ofstream out("bead_eigen_vectors.txt");
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
        Eigen::SelfAdjointEigenSolver<StiffnessMatrix> eigs(K);
        Eigen::MatrixXd eigen_vectors = eigs.eigenvectors().block(0, 0, K.rows(), nmodes);
        
        Eigen::VectorXd eigen_values = eigs.eigenvalues().segment(0, nmodes);
        std::cout << eigen_values << std::endl;
        std::ofstream out("bead_eigen_vectors.txt");
        out << eigen_vectors.rows() << " " << eigen_vectors.cols() << std::endl;
        for (int i = 0; i < eigen_vectors.rows(); i++)
        {
            for (int j = 0; j < eigen_vectors.cols(); j++)
                out << eigen_vectors(i, j) << " ";
            out << std::endl;
        }       
        out << std::endl;
        out.close();
    }

}



T HexFEMSolver::computeTotalEnergy(const VectorXT& u)
{
    T energy = 0.0;
    
    VectorXT projected = u;
    if (!run_diff_test)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            projected[offset] = target;
        });
    }
    deformed = undeformed + projected;

    if (plain_strain)
        addPlainStrainElastsicPotential(energy);
    else
        addNHElastsicPotential(energy);

    T penalty_energy = 0.0;
    iterateBCPenaltyPairs([&](int offset, T target)
    {
        penalty_energy += penalty_weight * 0.5 * std::pow(deformed[offset] - target, 2);
    });
    energy += penalty_energy;

    energy -= u.dot(f);
    return energy;
}


T HexFEMSolver::computeTotalVolume()
{
    T volume = 0.0;
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        volume += computeHexFEMVolume(x.transpose(), X.transpose());
    });
    return volume;
}


void HexFEMSolver::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}



void HexFEMSolver::computeElementDeformationGradient3D()
{
   auto shapeFunction = [&](const TV& xi)
    {
        Vector<T, 8> basis;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                    basis[4*(i-1) + 2 * (j-1) + k - 1] = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
        return basis;
    };

    auto dNdxi = [&](const TV& xi)
    {
        Matrix<T, 8, 3> dN_dxi;
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                {
                    int basis_id = 4*(i-1) + 2 * (j-1) + k - 1;
                    dN_dxi(basis_id, 0) = ((pow(-1.0, i)) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 1) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        (pow(-1.0, j)) * ((1.0) + pow(-1.0, k) * xi[2]) ) / 8.0;
                    dN_dxi(basis_id, 2) = (((1.0) + pow(-1.0, i) * xi[0]) * 
                        ((1.0) + pow(-1.0, j) * xi[1]) * (pow(-1.0, k)) ) / 8.0;
                }
        return dN_dxi;
    };
    T energy = 0.0;
    T lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
    T mu = E / 2.0 / (1.0 + nu);
    iterateHexElementSerial([&](int cell_idx){
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        HexNodes x = getHexNodesDeformed(cell_idx);
        HexNodes X = getHexNodesUndeformed(cell_idx);
        

        for (int i = 0; i < 8; i++)
            x.row(i) = min_corner.transpose() + 1.1 * (x.row(i) - min_corner.transpose());


        
        for (int i = 1; i < 3; i++)
            for (int j = 1; j < 3; j++)
                for (int k = 1; k < 3; k++)
                {
                    TV xi(pow(-1.0, i) / sqrt(3.0), pow(-1.0, j) / sqrt(3.0), pow(-1.0, k) / sqrt(3.0));
                    Matrix<T, 8, 3> dNdb = dNdxi(xi);
                    
                    TM dXdb = X.transpose() * dNdb;
                    TM dxdb = x.transpose() * dNdb;
                    TM defGrad = dxdb * dXdb.inverse();
                    
                    // std::cout << defGrad << std::endl;

                    TM greenStrain = 0.5 * (defGrad.transpose() * defGrad - TM::Identity());
                    auto trace = [&](const TM& matrix)
                    {
                        return matrix(0, 0) + matrix(1, 1) + matrix(2, 2);
                    };

                    energy += 0.5 * lambda * trace(greenStrain) * trace(greenStrain) + mu * trace(greenStrain*greenStrain);

                    // std::getchar();

                }
    });
    std::cout <<"E: " <<  energy << std::endl;
}


T HexFEMSolver::computeResidual(const VectorXT& u,
                                        VectorXT& residual)
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

    if (plain_strain)
        addPlainStrainElasticForceEntries(residual);
    else
        addNHElasticForceEntries(residual);

    iterateBCPenaltyPairs([&](int offset, T target)
    {
        residual[offset] -= penalty_weight * (deformed[offset] - target);
    });

    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
        
    return residual.norm();
}


bool HexFEMSolver::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    Timer t(true);
    Eigen::CholmodSupernodalLLT<StiffnessMatrix, Eigen::Lower> solver;
    // Eigen::PardisoLLT<StiffnessMatrix, Eigen::Lower> solver;
    T alpha = 1e-6;
    StiffnessMatrix H(K.rows(), K.cols());
    H.setIdentity(); H.diagonal().array() = 1e-12;
    K += H;
    solver.analyzePattern(K);
    
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
        
        // bool solve_success = true;
        // bool solve_success = (K * du - residual).norm() / residual.norm() < 1e-6;
        bool solve_success = du.norm() < 1e3;
        
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


void HexFEMSolver::buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K)
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
    
    if (plain_strain)
        addPlainStrainElasticHessianEntries(entries, false);
    else    
        addNHElasticHessianEntries(entries, false);

    iterateBCPenaltyPairs([&](int offset, T target)
    {
        entries.push_back(Entry(offset, offset, penalty_weight));
    });

    K.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();

}

bool HexFEMSolver::staticSolveStep(int step)
{
    if (step == 0)
    {
        iterateDirichletDoF([&](int offset, T target)
        {
            f[offset] = 0;
        });
        
    }

    VectorXT residual(deformed.rows());
    residual.setZero();

    T residual_norm = computeResidual(u, residual);
    std::cout << "[NEWTON] iter " << step << "/" << max_newton_iter << ": residual_norm " << residual_norm << " tol: " << newton_tol << std::endl;
    if (residual_norm < newton_tol)
    {
        checkGreenStrain();
        T e;
        addPlainStrainElastsicPotential(e);
        std::cout << "elastic potential " << e << std::endl;
        return true;
    }
    
    T dq_norm = lineSearchNewton(u, residual);

    if(step == max_newton_iter || dq_norm > 1e10 || dq_norm < 1e-12)
    {
        return true;
    }
    
    return false;
}


bool HexFEMSolver::staticSolve()
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
        
        if (verbose)
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

    std::cout << "# of newton solve: " << cnt << " exited with |g|: " << residual_norm << "|dq|: " << dq_norm  << std::endl;
    // std::cout << u.norm() << std::endl;
    if (cnt == max_newton_iter || dq_norm > 1e10 || residual_norm > 1)
        return false;
    return true;
    
}


T HexFEMSolver::lineSearchNewton(VectorXT& u, VectorXT& residual)
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
    T alpha = computeInversionFreeStepsize(u, du);
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


void HexFEMSolver::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    deformed = undeformed + u;

    int n_vtx = deformed.size() / 3;
    V.resize(n_vtx, 3);
    tbb::parallel_for(0, n_vtx, [&](int i)
    {
        V.row(i) = deformed.template segment<3>(i * 3).template cast<double>();
    });
    int n_faces = surface_indices.size() / 4 * 2;
    F.resize(n_faces, 3);
    C.resize(n_faces, 3);
    tbb::parallel_for(0, (int)surface_indices.size()/4, [&](int i)
    {
        F.row(i * 2 + 0) = Eigen::Vector3i(surface_indices[i * 4 + 2], 
                                            surface_indices[i * 4 + 1],
                                            surface_indices[i * 4 + 0]);
        F.row(i * 2 + 1) = Eigen::Vector3i(surface_indices[i * 4 + 3], 
                                            surface_indices[i * 4 + 2],
                                            surface_indices[i * 4 + 0]);
        C.row(i * 2 + 0) = Eigen::Vector3d(0, 0.3, 1);
        C.row(i * 2 + 1) = Eigen::Vector3d(0, 0.3, 1);
    });
}


void HexFEMSolver::buildGrid3D(const TV& _min_corner, const TV& _max_corner, T dx)
{
    min_corner = _min_corner;
    max_corner = _max_corner;

   vol = std::pow(dx, 3);
    std::vector<TV> nodal_position;
    for (T x = min_corner[0]; x < max_corner[0] + 0.1 * dx; x += dx)
    {
        for (T y = min_corner[1]; y < max_corner[1] + 0.1 * dx; y += dx)
        {
            for (T z = min_corner[2]; z < max_corner[2] + 0.1 * dx; z += dx)
            {
                nodal_position.push_back(TV(x, y, z));
            }
        }
    }
    deformed.resize(nodal_position.size() * 3);
    tbb::parallel_for(0, (int)nodal_position.size(), [&](int i)
    {
        for (int d = 0; d < 3; d++)
            deformed[i * 3 + d] = nodal_position[i][d];
    });

    undeformed = deformed;

    num_nodes = deformed.rows() / 3;

    f = VectorXT::Zero(deformed.rows());
    u = VectorXT::Zero(deformed.rows());

    int nx = std::floor((max_corner[0] - min_corner[0]) / dx) + 1;
    int ny = std::floor((max_corner[1] - min_corner[1]) / dx) + 1;
    int nz = std::floor((max_corner[2] - min_corner[2]) / dx) + 1;
    
    scene_range = IV(nx, ny, nz);
    // std::cout << "# nodes" < nodal_position.size() << std::endl;
    std::cout << "#nodes : " << scene_range.transpose() << std::endl;

    indices.resize((nx-1) * (ny-1) * (nz-1) * 8);
    surface_indices.resize(
        (nx - 1) * (nz - 1) * 4 * 2 + (nx - 1) * (ny - 1) * 4 * 2 + (ny - 1) * (nz - 1) * 4 * 2
    );
    surface_indices.setZero();
    indices.setZero();
    
    int cnt = 0;
    int surface_cnt = 0;
    for (int i = 0; i < nx - 1; i++)
    {
        for (int j = 0; j < ny - 1; j++)
        {
            for (int k = 0; k < nz - 1; k++)
            {
                Vector<int, 8> idx;

                idx[0] = globalOffset(IV(i, j, k));
                idx[1] = globalOffset(IV(i, j, k+1));
                idx[2] = globalOffset(IV(i, j+1, k));
                idx[3] = globalOffset(IV(i, j+1, k+1));
                idx[4] = globalOffset(IV(i+1, j, k));
                idx[5] = globalOffset(IV(i+1, j, k+1));
                idx[6] = globalOffset(IV(i+1, j+1, k));
                idx[7] = globalOffset(IV(i+1, j+1, k+1));


                indices.template segment<8>(cnt*8) = idx;
                cnt ++;
                if (i == 0)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i, j + 1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i, j + 1, k + 1));                        
                    surface_cnt++;
                }
                if (i == nx - 2)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i+1, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i+1, j, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j + 1, k));                        
                    surface_cnt++;
                }
                if ( k == 0 )
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i+1, j, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i, j + 1, k));                        
                    surface_cnt++;
                }
                if ( k == nz - 2 )
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j+1, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j, k+1));                        
                    surface_cnt++;
                }
                if (j == 0)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j, k));
                    surface_cnt++;
                }
                if (j == ny - 2)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j+1, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j+1, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j+1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j+1, k+1));
                    surface_cnt++;
                }
            }
        }
        
    }
}


void HexFEMSolver::createSceneFromNodes(const TV& _min_corner, 
    const TV& _max_corner, T dx, const std::vector<TV>& nodal_position)
{
    deformed.resize(nodal_position.size() * 3);
    tbb::parallel_for(0, (int)nodal_position.size(), [&](int i)
    {
        for (int d = 0; d < 3; d++)
            deformed[i * 3 + d] = nodal_position[i][d];
    });

    undeformed = deformed;

    num_nodes = deformed.rows() / 3;

    f = VectorXT::Zero(deformed.rows());
    u = VectorXT::Zero(deformed.rows());

    int nx = std::floor((max_corner[0] - min_corner[0]) / dx) + 1;
    int ny = std::floor((max_corner[1] - min_corner[1]) / dx) + 1;
    int nz = std::floor((max_corner[2] - min_corner[2]) / dx) + 1;
    
    scene_range = IV(nx, ny, nz);
    // std::cout << "# nodes" < nodal_position.size() << std::endl;
    std::cout << "#nodes : " << scene_range.transpose() << std::endl;

    indices.resize((nx-1) * (ny-1) * (nz-1) * 8);
    surface_indices.resize(
        (nx - 1) * (nz - 1) * 4 * 2 + (nx - 1) * (ny - 1) * 4 * 2 + (ny - 1) * (nz - 1) * 4 * 2
    );
    surface_indices.setZero();
    indices.setZero();
    
    int cnt = 0;
    int surface_cnt = 0;
    for (int i = 0; i < nx - 1; i++)
    {
        for (int j = 0; j < ny - 1; j++)
        {
            for (int k = 0; k < nz - 1; k++)
            {
                Vector<int, 8> idx;

                idx[0] = globalOffset(IV(i, j, k));
                idx[1] = globalOffset(IV(i, j, k+1));
                idx[2] = globalOffset(IV(i, j+1, k));
                idx[3] = globalOffset(IV(i, j+1, k+1));
                idx[4] = globalOffset(IV(i+1, j, k));
                idx[5] = globalOffset(IV(i+1, j, k+1));
                idx[6] = globalOffset(IV(i+1, j+1, k));
                idx[7] = globalOffset(IV(i+1, j+1, k+1));


                indices.template segment<8>(cnt*8) = idx;
                cnt ++;
                if (i == 0)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i, j + 1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i, j + 1, k + 1));                        
                    surface_cnt++;
                }
                if (i == nx - 2)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i+1, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i+1, j, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j + 1, k));                        
                    surface_cnt++;
                }
                if ( k == 0 )
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i+1, j, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i, j + 1, k));                        
                    surface_cnt++;
                }
                if ( k == nz - 2 )
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j+1, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j + 1, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j, k+1));                        
                    surface_cnt++;
                }
                if (j == 0)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j, k));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j, k+1));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j, k+1));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j, k));
                    surface_cnt++;
                }
                if (j == ny - 2)
                {
                    surface_indices[surface_cnt * 4 + 0] = globalOffset(IV(i, j+1, k+1));
                    surface_indices[surface_cnt * 4 + 1] = globalOffset(IV(i, j+1, k));
                    surface_indices[surface_cnt * 4 + 2] = globalOffset(IV(i+1, j+1, k));
                    surface_indices[surface_cnt * 4 + 3] = globalOffset(IV(i+1, j+1, k+1));
                    surface_cnt++;
                }
            }
        }
        
    }
}

void HexFEMSolver::setBCBendCorner(T curvature, T bending_direction)
{
    TV center = 0.5 * (max_corner + min_corner);

    penalty_pairs.clear();
    TV K1_dir(std::cos(bending_direction), 0.0, -std::sin(bending_direction));

    TV K2_dir = K1_dir.cross(TV(0, 1, 0)).normalized();

    T radius = 1.0 / curvature;

    TV cylinder_center = center - TV(0, radius, 0);

    iterateDirichletVertices([&](const TV& vtx, int idx)
    {
        TV d = vtx - center;
        T distance_along_cylinder_dir = d.dot(K1_dir);
        T distance_along_unwrapped_plane = d.dot(K2_dir);
        // unwrap cylinder to xy plane
        T arc_central_angle = distance_along_unwrapped_plane / radius;

        TV pt_projected = cylinder_center + distance_along_cylinder_dir * K1_dir + 
            radius * (std::sin(arc_central_angle) * K2_dir + std::cos(arc_central_angle) * TV(0, 1, 0));
                
        for (int d = 0; d < 3; d++)
            penalty_pairs.push_back(std::make_pair(idx * 3 + d, pt_projected[d]));
    });
    
}

void HexFEMSolver::addCornerVtxToDirichletVertices(const Vector<bool, 4>& flag)
{
    T region = 1e-1;
    for (int i = 0; i < num_nodes; i++)
    {
        TV x = undeformed.segment<3>(i * 3);
        bool back_face = x[1] < min_corner[1] + 1e-6;
        if (!back_face)
            continue;
        if (flag[0]) // bottom left
            if (x[0] < min_corner[0] + region && x[2] < min_corner[2] + region)
                dirichlet_vertices.push_back(i);
        if (flag[1]) // bottom right
            if (x[0] > max_corner[0] - region && x[2] < min_corner[2] + region)
                dirichlet_vertices.push_back(i);
        if (flag[2]) // top right
            if (x[0] > max_corner[0] - region && x[2] > max_corner[2] - region)
                dirichlet_vertices.push_back(i);
        if (flag[3]) // top left
            if (x[0] < min_corner[0] + region && x[2] > max_corner[2] - region)
                dirichlet_vertices.push_back(i);
    }
}

void HexFEMSolver::computeBoundingBox()
{
    min_corner.setConstant(1e6);
    max_corner.setConstant(-1e6);

    for (int i = 0; i < num_nodes; i++)
    {
        for (int d = 0; d < 3; d++)
        {
            max_corner[d] = std::max(max_corner[d], undeformed[i * 3 + d]);
            min_corner[d] = std::min(min_corner[d], undeformed[i * 3 + d]);
        }
    }
}

void HexFEMSolver::penaltyInPlane(int dir, T percent)
{
    
    penalty_pairs.clear();
    
    T region = 0.01;
    for (int i = 0; i < num_nodes; i++)
    {
        T dx = max_corner[0] - min_corner[0];
        TV x = undeformed.segment<3>(i * 3);
        if (dir == 0)
        {
            if (x[0] < min_corner[0] + region * dx)
            {
                TV target = x + TV(-percent * dx, 0, 0);
                for (int d = 0; d < 3; d++)
                    penalty_pairs.push_back(std::make_pair(i * 3 + d, target[d]));
            }
            if (x[0] > max_corner[0] - region * dx)
            {
                TV target = x + TV(percent * dx, 0.0, 0.0);
                for (int d = 0; d < 3; d++)
                    penalty_pairs.push_back(std::make_pair(i * 3 + d, target[d]));
            }
        }
        else if (dir == 1)
        {
            T dz = max_corner[2] - min_corner[2];
            if (x[2] < min_corner[2] + region * dz)
            {
                TV target = x + TV(0, 0, -percent * dz);
                for (int d = 0; d < 3; d++)
                    penalty_pairs.push_back(std::make_pair(i * 3 + d, target[d]));
            }
            if (x[2] > max_corner[2] - region * dz)
            {
                TV target = x + TV(1e-3, 1e-3, percent * dz);
                for (int d = 0; d < 3; d++)
                    penalty_pairs.push_back(std::make_pair(i * 3 + d, target[d]));
            }
        }
    }
}
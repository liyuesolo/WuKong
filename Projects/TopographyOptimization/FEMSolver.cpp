#include "FEMSolver.h"
#include "autodiff/HexFEMNeoHookean3D.h"

// #define USE_CHOLMOD

#ifdef USE_CHOLMOD
    #include "/home/yueli/Documents/ETH/WuKong/Solver/CHOLMODSolver.hpp"
#endif

template<class T, int dim>
T FEMSolver<T, dim>::computeTotalEnergy(const VectorXT& u)
{
    T energy = 0.0;
    if constexpr (dim == 3)
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

        iterateHexElementSerial([&](int cell_idx){
            HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
            HexNodes x = getHexNodesDeformed(cell_idx);
            HexNodes X = getHexNodesUndeformed(cell_idx);
            energy += computeHexFEMNeoHookeanEnergy<T>(lambda, mu, x.transpose(), X.transpose());
        });
        energy -= u.dot(f);
    }
    return energy;
}

template<class T, int dim>
void FEMSolver<T, dim>::projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data)
{
    for (auto iter : data)
    {
        A.row(iter.first) *= 0.0;
        A.col(iter.first) *= 0.0;
        A.coeffRef(iter.first, iter.first) = 1.0;
    }

}


template<class T, int dim>
void FEMSolver<T, dim>::computeElementDeformationGradient3D()
{
    if constexpr (dim == 3)
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
            Matrix<T, 8, dim> dN_dxi;
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
            
            // TV element_center = TV::Zero();
            // for (int i = 0; i < 8; i++)
            //     element_center += x.row(i) / 8.0;
            
            // for (int i = 0; i < 8; i++)
            //     x.row(i) = element_center.transpose() + 1.1 * (x.row(i) - element_center.transpose());

            for (int i = 0; i < 8; i++)
                x.row(i) = min_corner.transpose() + 1.1 * (x.row(i) - min_corner.transpose());


            
            for (int i = 1; i < 3; i++)
                for (int j = 1; j < 3; j++)
                    for (int k = 1; k < 3; k++)
                    {
                        TV xi(pow(-1.0, i) / sqrt(3.0), pow(-1.0, j) / sqrt(3.0), pow(-1.0, k) / sqrt(3.0));
                        Matrix<T, 8, dim> dNdb = dNdxi(xi);
                        
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
}

template<class T, int dim>
T FEMSolver<T, dim>::computeResidual(const VectorXT& u,
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

    if constexpr (dim == 3)
    {
        
        iterateHexElementSerial([&](int cell_idx){
            HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
            
            // std::cout << nodal_indices.transpose() << std::endl;

            HexNodes x = getHexNodesDeformed(cell_idx);
            HexNodes X = getHexNodesUndeformed(cell_idx);

            
            Vector<T, 24> F;
            computeHexFEMNeoHookeanEnergyGradient<T>(lambda, mu, x.transpose(), X.transpose(), F);

            F *= -1.0;
            for (int i = 0; i < nodal_indices.size(); i++)
            {
                int node_i = nodal_indices[i];
                for (int d = 0; d < dim; d++)
                {
                    residual[node_i * dim + d] += F[i * dim + d];
                }
            }
            
        });
    }
    // std::getchar();
    if (!run_diff_test)
        iterateDirichletDoF([&](int offset, T target)
        {
            residual[offset] = 0;
        });
        
    return residual.norm();
}

template<class T, int dim>
bool FEMSolver<T, dim>::linearSolve(StiffnessMatrix& K, 
    VectorXT& residual, VectorXT& du)
{
    StiffnessMatrix I(K.rows(), K.cols());
    I.setIdentity();

    StiffnessMatrix H = K;

#ifdef USE_CHOLMOD
    Noether::CHOLMODSolver<typename StiffnessMatrix::StorageIndex> solver;
    solver.set_pattern(K);
    
    solver.analyze_pattern();
    solver.factorize();
    
    solver.solve(residual.data(), du.data(), true);
#else
    
    Eigen::SimplicialLDLT<StiffnessMatrix> solver;
    // Eigen::SimplicialLDLT<Eigen::SparseMatrix<T, Eigen::ColMajor, typename StiffnessMatrix::StorageIndex>> solver;
    
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

        VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;

        for (int i = 0; i < d_vector.size(); i++)
        {
            if (d_vector[i] < 0)
            {
                num_negative_eigen_values++;
                break;
            }
        
        }
        bool positive_definte = num_negative_eigen_values == 0;
        bool search_dir_correct_sign = dot_dx_g > 1e-6;
        bool solve_success = (K*du - residual).norm() < 1e-6 && solver.info() == Eigen::Success;

        if (positive_definte && search_dir_correct_sign && solve_success)
            break;
        else
        {
            K = H + mu * I;        
            mu *= 10;
        }
        // std::cout << i << std::endl;
    }
#endif
    return true;
}

template<class T, int dim>
void FEMSolver<T, dim>::buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K)
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
    if constexpr (dim == 3)
    {
        iterateHexElementSerial([&](int cell_idx){
            Vector<int, 8> nodal_indices = indices.segment<8>(cell_idx * 8);
            HexNodes x = getHexNodesDeformed(cell_idx);
            HexNodes X = getHexNodesUndeformed(cell_idx);
            Matrix<T, 24, 24> J;
            computeHexFEMNeoHookeanEnergyHessian<T>(lambda, mu, x.transpose(), X.transpose(), J);
            
            for (int i = 0; i < nodal_indices.size(); i++)
            {
                int node_i = nodal_indices[i];
                for (int j = 0; j < nodal_indices.size(); j++)
                {
                    int node_j = nodal_indices[j];
                    for (int d = 0; d < dim; d++)
                        for (int dd = 0; dd < dim; dd++)
                            entries.push_back(Entry(node_i * dim + d, node_j * dim + dd, J(i * dim + d, j * dim + dd)));
                }
            }
        });
    }

    K.setFromTriplets(entries.begin(), entries.end());

    if (!run_diff_test)
        projectDirichletDoFMatrix(K, dirichlet_data);
    
    K.makeCompressed();

}

template<class T, int dim>
bool FEMSolver<T, dim>::staticSolve()
{
    int cnt = 0;
    T residual_norm = 1e10, dq_norm = 1e10;

    int max_newton_iter = 1000;

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

template<class T, int dim>
T FEMSolver<T, dim>::lineSearchNewton(VectorXT& u, VectorXT& residual)
{
    VectorXT du = residual;
    du.setZero();

    StiffnessMatrix K(residual.rows(), residual.rows());
    buildSystemMatrix(u, K);
    bool success = linearSolve(K, residual, du);
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


// template class FEMSolver<float, 2>;
// template class FEMSolver<float, 3>;
// template class FEMSolver<double, 2>;
template class FEMSolver<double, 3>;
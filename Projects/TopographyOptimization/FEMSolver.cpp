// #include <filesystem>
// using std::filesystem::current_path;
#include <fstream>
#include "FEMSolver.h"
#include "autodiff/HexFEM.h"

#include "Timer.h"
#include <Eigen/PardisoSupport>

#include <Spectra/SymEigsShiftSolver.h>
#include <Spectra/MatOp/SparseSymShiftSolve.h>
#include <Spectra/SymEigsSolver.h>
#include <Spectra/MatOp/SparseSymMatProd.h>

template<class T, int dim>
void FEMSolver<T, dim>::computeInternalForce(const VectorXT& _u, VectorXT& dPsidu)
{

}

template<class T, int dim>
void FEMSolver<T, dim>::computedfdX(const VectorXT& u, StiffnessMatrix& dfdX)
{

}

template<class T, int dim>
void FEMSolver<T, dim>::computeEigenMode()
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

template<class T, int dim>
T FEMSolver<T, dim>::computeElasticPotential(const VectorXT& _u)
{
    std::cout << "not implemented yet" << std::endl;
    return 0.0;
}

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
            

            if (model == NeoHookean)
                energy += computeHexFEMNeoHookeanEnergy<T>(lambda, mu, x.transpose(), X.transpose());
            else if (model == StVK)
                energy += computeHexFEMStVKEnergy(lambda, mu, x.transpose(), X.transpose());
            else
                std::cout << "unknown consititutive model at " << __FILE__ << std::endl;

        });
        energy -= u.dot(f);
    }
    return energy;
}

template<class T, int dim>
T FEMSolver<T, dim>:: computeTotalVolume()
{
    T volume = 0.0;
    if constexpr (dim == 3)
    {
        iterateHexElementSerial([&](int cell_idx){
            HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
            HexNodes x = getHexNodesDeformed(cell_idx);
            HexNodes X = getHexNodesUndeformed(cell_idx);
            volume += computeHexFEMVolume(x.transpose(), X.transpose());
        });
    }
    return volume;
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
            if (model == NeoHookean)
                computeHexFEMNeoHookeanEnergyGradient<T>(lambda, mu, x.transpose(), X.transpose(), F);
            else if (model == StVK)
                computeHexFEMStVKEnergyGradient(lambda, mu, x.transpose(), X.transpose(), F);
            else
                std::cout << "unknown consititutive model at " << __FILE__ << std::endl;

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

        
        // VectorXT d_vector = solver.vectorD();
        int num_negative_eigen_values = 0;

        // for (int i = 0; i < d_vector.size(); i++)
        // {
        //     if (d_vector[i] < 0)
        //     {
        //         num_negative_eigen_values++;
        //         break;
        //     }
        
        // }
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
        if (i == 49)
            return false;
    }
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

            if (model == NeoHookean)
                computeHexFEMNeoHookeanEnergyHessian<T>(lambda, mu, x.transpose(), X.transpose(), J);
            else if (model == StVK)
                computeHexFEMStVKEnergyHessian(lambda, mu, x.transpose(), X.transpose(), J);
            else
                std::cout << "unknown consititutive model at " << __FILE__ << std::endl;

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

template<class T, int dim>
T FEMSolver<T, dim>::lineSearchNewton(VectorXT& u, VectorXT& residual)
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

template<class T, int dim>
void FEMSolver<T, dim>::generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
{
    if constexpr (dim == 3)
    {
        deformed = undeformed + u;

        int n_vtx = deformed.size() / dim;
        V.resize(n_vtx, 3);
        tbb::parallel_for(0, n_vtx, [&](int i)
        {
            V.row(i) = deformed.template segment<dim>(i * dim).template cast<double>();
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

        // std::cout << V << std::endl;
        // std::cout << F << std::endl;
    }
}

template<class T, int dim>
void FEMSolver<T, dim>::buildGrid3D(const TV& _min_corner, const TV& _max_corner, T dx)
{
    min_corner = _min_corner;
    max_corner = _max_corner;

    if constexpr (dim == 3)
    {
        vol = std::pow(dx, dim);
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
        deformed.resize(nodal_position.size() * dim);
        tbb::parallel_for(0, (int)nodal_position.size(), [&](int i)
        {
            for (int d = 0; d < dim; d++)
                deformed[i * dim + d] = nodal_position[i][d];
        });

        undeformed = deformed;

        num_nodes = deformed.rows() / dim;

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
        // std::cout << "done" << std::endl;
    }   
}

template<class T, int dim>
void FEMSolver<T, dim>::createSceneFromNodes(const TV& _min_corner, 
    const TV& _max_corner, T dx, const std::vector<TV>& nodal_position)
{
    model = StVK;

    if constexpr (dim == 3)
    {
        deformed.resize(nodal_position.size() * dim);
        tbb::parallel_for(0, (int)nodal_position.size(), [&](int i)
        {
            for (int d = 0; d < dim; d++)
                deformed[i * dim + d] = nodal_position[i][d];
        });

        undeformed = deformed;

        num_nodes = deformed.rows() / dim;

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
}

template<class T, int dim>
void FEMSolver<T, dim>::loadFromMesh(std::string filename)
{
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;


    
}

// template class FEMSolver<float, 2>;
// template class FEMSolver<float, 3>;
// template class FEMSolver<double, 2>;
template class FEMSolver<double, 3>;
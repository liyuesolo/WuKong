#ifndef FEM_SOLVER_H
#define FEM_SOLVER_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"


template<class T, int dim>
class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;
    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    // typedef long StorageIndex;
    // using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using HexNodes = Matrix<T, 8, dim>;
    using HexIdx = Vector<int, 8>;

    using Entry = Eigen::Triplet<T>;

    VectorXT u;
    VectorXT deformed, undeformed;
    VectorXi indices;
    VectorXi surface_indices;
    VectorXT f;

    std::unordered_map<int, T> dirichlet_data;

    
    int num_nodes;   
    bool verbose = false;
    bool run_diff_test = false;
    
    // simulation-related data
    T vol = 1.0;
    T E = 1e7;
    T nu = 0.42;

    T lambda, mu;

    T newton_tol = 1e-6;
    
    //scene related data
    TV min_corner, max_corner;
    IV scene_range;

public:
    FEMSolver() {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }
    ~FEMSolver() {}


    // ################## Boundary Conditions ##################

    void addDirichletLambda(std::function<bool(const TV&, TV&)> node_helper);
    void addNeumannLambda(std::function<bool(const TV&, TV&)> node_helper, VectorXT& f);
    void fixAxisEnd(int axis);
    
    void computeElementDeformationGradient3D();

    // helper

    HexNodes getHexNodesDeformed(int cell_idx)
    {
        HexNodes cellx;
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        for (int i = 0; i < 8; i++)
        {
            cellx.row(i) = deformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return cellx;
    }

    HexNodes getHexNodesUndeformed(int cell_idx)
    {
        HexNodes cellX;
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        for (int i = 0; i < 8; i++)
        {
            cellX.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return cellX;
    }   


    // ###################### iterators ######################

    template <typename OP>
    void iterateHexElementSerial(const OP& f)
    {
        for (int i = 0; i < int(indices.size()/8); i++)
            f(i);
    }

    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    void initialize()
    {

    }

    T computeTotalEnergy(const VectorXT& u);

    void buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K);
    
    T computeResidual(const VectorXT& u,  VectorXT& residual);

    T lineSearchNewton(VectorXT& u,  VectorXT& residual);

    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    void reset()
    {

    }

    inline int globalOffset(const IV& node_offset)
    {
        if constexpr (dim == 2)
        {
            return node_offset[0] * scene_range[1] + node_offset[1];
        }
        else if constexpr (dim == 3)
        {
            return node_offset[0] * scene_range[1] * scene_range[2] + node_offset[1] * scene_range[2] + node_offset[2];
        }
    }

    // DerivativeTest.cpp
    void derivativeTest();
    void checkTotalGradient();
    void checkTotalHessian();


    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C)
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

// Scene.cpp
    void buildGrid3D(const TV& _min_corner, const TV& _max_corner, T dx)
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

                        // idx[0] = globalOffset(IV(i, j, k));
                        // idx[1] = globalOffset(IV(i+1, j, k));
                        // idx[2] = globalOffset(IV(i+1, j+1, k));
                        // idx[3] = globalOffset(IV(i, j+1, k));
                        // idx[4] = globalOffset(IV(i, j, k+1));
                        // idx[5] = globalOffset(IV(i+1, j, k+1));
                        // idx[6] = globalOffset(IV(i+1, j+1, k+1));
                        // idx[7] = globalOffset(IV(i, j+1, k+1));

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
    void loadQuadMesh() {}
};


#endif
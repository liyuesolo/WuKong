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

class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, 3>;
    using TM = Matrix<T, 3, 3>;

    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

    using TetNodes = Matrix<T, 4, 3>;
    using TetIdx = Vector<int, 4>;

    using Face = Vector<int, 3>;

public:
    int dim = 3;

    VectorXT u;
    VectorXT deformed, undeformed;
    VectorXi indices;
    VectorXi surface_indices;

    std::unordered_map<int, T> dirichlet_data;

    int num_nodes;   
    int num_ele;
    int num_surface_faces;

    bool verbose = false;
    bool run_diff_test = false;


    T vol = 1.0;
    T E = 1e6;
    T density = 7.85e4; 
    T nu = 0.42;
    

    T lambda, mu;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;

    TV min_corner, max_corner;

public:

    // ###################### iterators ######################
    template <typename OP>
    void iterateTetsSerial(const OP& f)
    {
        for (int i = 0; i < int(indices.size()/4); i++)
        {
            TetIdx tet_idx = indices.segment<4>(i * 4);
            TetNodes tet_deformed = getTetNodesDeformed(tet_idx);
            TetNodes tet_undeformed = getTetNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateTetsParallel(const OP& f)
    {
        tbb::parallel_for(0, int(indices.size()/4), [&](int i)
        {
            TetIdx tet_idx = indices.segment<4>(i * 4);
            TetNodes tet_deformed = getTetNodesDeformed(tet_idx);
            TetNodes tet_undeformed = getTetNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        });
    }

    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }
    

public:
    template<int size>
    void addForceEntry(VectorXT& residual, 
        const VectorXi& vtx_idx, 
        const Vector<T, size>& gradent)
    {
        if (vtx_idx.size() * 3 != size)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.segment<3>(vtx_idx[i] * 3) += gradent.template segment<3>(i * 3);
    }

    template<int size>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const VectorXi& vtx_idx, 
        const Matrix<T, size, size>& hessian)
    {
        if (vtx_idx.size() * 3 != size)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < 3; k++)
                    for (int l = 0; l < 3; l++)
                    {
                        if (std::abs(hessian(i * 3 + k, j * 3 + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * 3 + k, dof_j * 3 + l, hessian(i * 3 + k, j * 3 + l)));                
                    }
            }
        }
    }

    TetNodes getTetNodesDeformed(const TetIdx& nodal_indices)
    {
        if (nodal_indices.size() != 4)
            std::cout << "getTetNodesDeformed() not a tet" << std::endl; 
        TetNodes tet_x;
        for (int i = 0; i < 4; i++)
        {
            tet_x.row(i) = deformed.segment<3>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    TetNodes getTetNodesUndeformed(const TetIdx& nodal_indices)
    {
        if (nodal_indices.size() != 4)
            std::cout << "getTetNodesUndeformed() not a tet" << std::endl;
        TetNodes tet_x;
        for (int i = 0; i < 4; i++)
        {
            tet_x.row(i) = undeformed.segment<3>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    // Scene.cpp
    void initializeElementData(const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    // FEMSolver.cpp
    T computeTotalEnergy(const VectorXT& u);

    void buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K);

    T computeResidual(const VectorXT& u,  VectorXT& residual);

    T lineSearchNewton(VectorXT& u,  VectorXT& residual);

    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    // DerivativeTest.cpp
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    //Helper.cpp
    void saveTetOBJ(const std::string& filename, const TetNodes& tet_vtx);

    FEMSolver() {}
    ~FEMSolver() {}

};

#endif

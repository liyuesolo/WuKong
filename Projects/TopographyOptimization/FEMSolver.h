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
    typedef int StorageIndex;
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

    enum ConstitutiveModel
    {
        NeoHookean, StVK
    };

    ConstitutiveModel model = NeoHookean;
    
    int num_nodes;   
    bool verbose = false;
    bool run_diff_test = false;
    
    // simulation-related data
    T vol = 1.0;
    T E = 221.88 * 1e9;
    T density = 7.85e4; 
    T nu = 0.3;
    T dx = 1e-3;

    T lambda, mu;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;
    
    std::string name = "HexFEM";
    
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

    
    T computeElasticPotential(const VectorXT& _u);
    
    T computeTotalEnergy(const VectorXT& u);

    void buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K);

    void computedfdX(const VectorXT& u, StiffnessMatrix& dfdX);
    
    void computeInternalForce(const VectorXT& _u, VectorXT& dPsidu);
    
    T computeResidual(const VectorXT& u,  VectorXT& residual);

    T lineSearchNewton(VectorXT& u,  VectorXT& residual);

    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    void reset()
    {

    }

    void updateRestshape() {}

    T computeTotalVolume();

    void loadFromMesh(std::string filename);
    
    void computeEigenMode();

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
    void checkdfdX();


    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void buildGrid3D(const TV& _min_corner, const TV& _max_corner, T dx);

    void createSceneFromNodes(const TV& _min_corner, const TV& _max_corner, T dx, 
        const std::vector<TV>& nodal_position);

    void loadQuadMesh() {}
};


#endif
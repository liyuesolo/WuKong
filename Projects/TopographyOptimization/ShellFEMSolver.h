#ifndef SHELL_FEM_SOLVER_H
#define SHELL_FEM_SOLVER_H
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
class ShellFEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using IV2 = Vector<int, 2>;
    using TM = Matrix<T, dim, dim>;
    using TM2 = Matrix<T, 2, 2>;
    using Hessian = Eigen::Matrix<T, dim * dim, dim * dim>;
    using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using Hinges = Matrix<int, Eigen::Dynamic, 4>;
    // typedef long StorageIndex;
    typedef int StorageIndex;
    // using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using FaceVtx = Matrix<T, 3, dim>;
    using FaceIdx = Vector<int, 3>;
    using HingeIdx = Vector<int, 4>;
    using HingeVtx = Matrix<T, 4, dim>;

    using Entry = Eigen::Triplet<T>;

    VectorXT u;
    VectorXT deformed, undeformed;
    VectorXi faces;
    VectorXT f;

    std::unordered_map<int, T> dirichlet_data;

    std::vector<TM2> Xinv;
    VectorXT rest_area;
    VectorXT thickness;

    
    Hinges hinges;

    int num_nodes = 0;   
    int max_newton_iter = 0;
    bool verbose = false;
    bool run_diff_test = false;
    bool add_bending = true;
    bool add_stretching = true;
    bool gravitional_energy = true;
    
    std::string name = "ShellFEM";

    T E = 221.88 * 1e9;
    T nu = 0.3;

    T lambda, mu;

    T newton_tol = 1e-6;

    T density = 7.85e4; 

    TV gravity = TV::Zero();
    
    //scene related data
    TV min_corner, max_corner;
    IV scene_range;

public:
    ShellFEMSolver() {
        updateLameParameters();
        gravity[1] = 9.8;
    }

    ~ShellFEMSolver() {}

    void updateLameParameters()
    {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }


    HingeVtx getHingeVtxDeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = deformed.template segment<dim>(hi[i]*dim);
        }
        return cellx;
    }

    HingeVtx getHingeVtxUndeformed(const HingeIdx& hi)
    {
        HingeVtx cellx;
        for (int i = 0; i < 4; i++)
        {
            cellx.row(i) = undeformed.template segment<dim>(hi[i]*dim);
        }
        return cellx;
    }

    FaceVtx getFaceVtxDeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = deformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return cellx;
    }

    FaceVtx getFaceVtxUndeformed(int face)
    {
        FaceVtx cellx;
        FaceIdx nodal_indices = faces.segment<3>(face * 3);
        for (int i = 0; i < 3; i++)
        {
            cellx.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return cellx;
    }

    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <typename OP>
    void iterateFaceSerial(const OP& f)
    {
        for (int i = 0; i < faces.rows()/3; i++)
            f(i);
    }

    template <typename OP>
    void iterateHingeSerial(const OP& f)
    {
        for (int i = 0; i < hinges.rows(); i++)
        {
            const Vector<int, 4> nodes = hinges.row(i);
            f(nodes);   
        }
    }

    inline int globalOffset(const IV& node_offset)
    {
        
        return node_offset[0] * scene_range[2] + node_offset[2];
    }

    void buildHingeStructure();

    void computeRestShape();

    int nFaces () { return faces.rows() / 3; }

    T computeTotalEnergy(const VectorXT& _u);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    
    T computeResidual(const VectorXT& _u,  VectorXT& residual);

    T lineSearchNewton(VectorXT& _u,  VectorXT& residual, int ls_max = 15);

    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);

    void createSceneFromNodes(const TV& _min_corner, const TV& _max_corner, T dx, 
        const std::vector<TV>& nodal_position);

    void addDirichletLambda(std::function<bool(const TV&, TV&)> node_helper);
    void addNeumannLambda(std::function<bool(const TV&, TV&)> node_helper, VectorXT& f);
    void fixAxisEnd(int axis);

    void computeEigenMode();

    void derivativeTest();
    void checkTotalGradient();
    void checkTotalHessian();
    
};
#endif
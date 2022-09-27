#ifndef HEX_FEM_SOLVER_H
#define HEX_FEM_SOLVER_H

#include <utility>
#include <iostream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>

#include "VecMatDef.h"

class HexFEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using TV = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TM = Matrix<T, 3, 3>;
    // using Hessian = Eigen::Matrix<T, 3 * 3, 3 * 3>;
    // using TVStack = Matrix<T, dim, Eigen::Dynamic>;
    
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    // typedef long StorageIndex;
    typedef int StorageIndex;
    // using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using HexNodes = Matrix<T, 8, 3>;
    using HexIdx = Vector<int, 8>;

    using Entry = Eigen::Triplet<T>;

    VectorXT u;
    VectorXT deformed, undeformed;
    VectorXi indices;
    VectorXi surface_indices;
    VectorXT f;

    std::unordered_map<int, T> dirichlet_data;
    std::vector<std::pair<int, T>> penalty_pairs;
    std::vector<int> dirichlet_vertices;

    int num_nodes;   
    bool verbose = false;
    bool run_diff_test = false;
    bool plain_strain = false;
    bool stvk = false;
    
    // simulation-related data
    T vol = 1.0;
    T E = 2.6e7;
    T density = 7.85e4; 
    T nu = 0.48;
    T dx = 1e-3;
    T penalty_weight = 1e6;
    T KL_stiffness = 0.0;
    T KL_stiffness_shear = 0.0;

    T lambda, mu;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;
    
    
    //scene related data
    TV min_corner, max_corner;

    IV scene_range;

public:
    HexFEMSolver() {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }
    ~HexFEMSolver() {}

    void updateLameParams()
    {
        lambda = E * nu / (1.0 + nu) / (1.0 - 2.0 * nu);
        mu = E / 2.0 / (1.0 + nu);
    }

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
            cellx.row(i) = deformed.segment<3>(nodal_indices[i]*3);
        }
        return cellx;
    }

    HexNodes getHexNodesUndeformed(int cell_idx)
    {
        HexNodes cellX;
        HexIdx nodal_indices = indices.segment<8>(cell_idx * 8);
        for (int i = 0; i < 8; i++)
        {
            cellX.row(i) = undeformed.segment<3>(nodal_indices[i]*3);
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

    template <class OP>
    void iterateBCPenaltyPairs(const OP& f)
    {
        for (auto pair : penalty_pairs)
        {
            f(pair.first, pair.second);
        }
    }

    template <class OP>
    void iterateDirichletVertices(const OP& f) {
        for (auto vtx_idx : dirichlet_vertices){
            TV vtx = deformed.segment<3>(vtx_idx * 3);
            f(vtx, vtx_idx);
        } 
    }

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

private:
    inline T getSmallestPositiveRealQuadRoot(T a, T b, T c, T tol)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::sqrt;
        T t;
        if (abs(a) <= tol) {
            if (abs(b) <= tol) // f(x) = c > 0 for all x
                t = -1;
            else
                t = -c / b;
        }
        else {
            T desc = b * b - 4 * a * c;
            if (desc > 0) {
                t = (-b - sqrt(desc)) / (2 * a);
                if (t < 0)
                    t = (-b + sqrt(desc)) / (2 * a);
            }
            else // desv<0 ==> imag
                t = -1;
        }
        return t;
    }

    inline T getSmallestPositiveRealCubicRoot(T a, T b, T c, T d, T tol = 1e-10)
    {
        // return negative value if no positive real root is found
        using std::abs;
        using std::complex;
        using std::pow;
        using std::sqrt;
        T t = -1;
        if (abs(a) <= tol)
            t = getSmallestPositiveRealQuadRoot(b, c, d, tol);
        else {
            complex<T> i(0, 1);
            complex<T> delta0(b * b - 3 * a * c, 0);
            complex<T> delta1(2 * b * b * b - 9 * a * b * c + 27 * a * a * d, 0);
            complex<T> C = pow((delta1 + sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            if (abs(C) < tol)
                C = pow((delta1 - sqrt(delta1 * delta1 - 4.0 * delta0 * delta0 * delta0)) / 2.0, 1.0 / 3.0);
            complex<T> u2 = (-1.0 + sqrt(3.0) * i) / 2.0;
            complex<T> u3 = (-1.0 - sqrt(3.0) * i) / 2.0;
            complex<T> t1 = (b + C + delta0 / C) / (-3.0 * a);
            complex<T> t2 = (b + u2 * C + delta0 / (u2 * C)) / (-3.0 * a);
            complex<T> t3 = (b + u3 * C + delta0 / (u3 * C)) / (-3.0 * a);
            if ((abs(imag(t1)) < tol) && (real(t1) > 0))
                t = real(t1);
            if ((abs(imag(t2)) < tol) && (real(t2) > 0) && ((real(t2) < t) || (t < 0)))
                t = real(t2);
            if ((abs(imag(t3)) < tol) && (real(t3) > 0) && ((real(t3) < t) || (t < 0)))
                t = real(t3);
        }
        return t;
    }
    
public:
    void initialize()
    {

    }

    
    T computeTotalEnergy(const VectorXT& u);

    void buildSystemMatrix(const VectorXT& u, StiffnessMatrix& K);

    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    
    T computeResidual(const VectorXT& u,  VectorXT& residual);

    T lineSearchNewton(VectorXT& u,  VectorXT& residual);

    bool staticSolveStep(int step);
    
    bool staticSolve();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    void reset()
    {
        deformed = undeformed;
        u.setZero();
    }

    void checkGreenStrain();
    void checkHessianPD(bool save_txt = false);

    T computeTotalVolume();
    
    void computeEigenMode();

    inline int globalOffset(const IV& node_offset)
    {
        return node_offset[0] * scene_range[1] * scene_range[2] + node_offset[1] * scene_range[2] + node_offset[2];
    }

    // DerivativeTest.cpp
    void derivativeTest();
    void checkTotalGradient();
    void checkTotalHessian();

    void addPlainStrainElastsicPotential(T& energy);
    void addPlainStrainElasticForceEntries(VectorXT& residual);
    void addPlainStrainElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    void addNHElastsicPotential(T& energy);
    void addNHElasticForceEntries(VectorXT& residual);
    void addNHElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void buildGrid3D(const TV& _min_corner, const TV& _max_corner, T dx);
    void computeBoundingBox();
    void createSceneFromNodes(const TV& _min_corner, const TV& _max_corner, T dx, 
        const std::vector<TV>& nodal_position);

    void setBCBendCorner(T curvature, T bending_direction);
    void addCornerVtxToDirichletVertices(const Vector<bool, 4>& flag);
    void penaltyInPlane(int dir, T percent);
};


#endif
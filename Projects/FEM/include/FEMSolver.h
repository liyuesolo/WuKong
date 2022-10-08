#ifndef FEM_SOLVER_H
#define FEM_SOLVER_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_map>
#include <complex>

#include "VecMatDef.h"

using Eigen::MatrixXd;
using Eigen::MatrixXi;

template<int dim>
class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, dim>;
    using IV = Vector<int, dim>;
    using TM = Matrix<T, dim, dim>;

    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    

    using Entry = Eigen::Triplet<T>;

    using EleNodes = Matrix<T, dim + 1, dim>;
    using EleIdx = Vector<int, dim + 1>;

    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

public:
    
    VectorXT u;
    VectorXT f;
    VectorXT deformed, undeformed;
    VectorXi indices;
    VectorXi surface_indices;

    VectorXT residual_step;

    std::unordered_map<int, T> dirichlet_data;

    std::vector<std::pair<int, T>> penalty_pairs;

    int num_nodes;   
    int num_ele;
    int num_surface_faces;

    bool verbose = false;
    bool run_diff_test = false;

    T vol = 1.0;
    T E = 2.6 * 1e8;
    T nu = 0.48;
    
    T penalty_weight = 1e6;
    bool use_penalty = false;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;

    TV min_corner, max_corner;
    TV center;
    
    bool project_block_PD = false;

    // IPC
    T max_barrier_weight = 1e8;
    bool add_friction = false;
    T friction_mu = 0.5;
    T epsv_times_h = 1e-5;
    bool self_contact = false;
    bool use_ipc = false;
    int num_ipc_vtx = 0;
    T barrier_distance = 1e-2;
    T barrier_weight = 1.0;
    T ipc_min_dis = 1e-6;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_faces;

public:

    // ###################### iterators ######################
    template <typename OP>
    void iterateElementSerial(const OP& f)
    {
        for (int i = 0; i < int(indices.size()/(dim + 1)); i++)
        {
            EleIdx tet_idx = indices.segment<dim + 1>(i * (dim + 1));
            EleNodes tet_deformed = getEleNodesDeformed(tet_idx);
            EleNodes tet_undeformed = getEleNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateElementParallel(const OP& f)
    {
        tbb::parallel_for(0, int(indices.size()/(dim + 1)), [&](int i)
        {
            EleIdx tet_idx = indices.segment<dim + 1>(i * (dim + 1));
            EleNodes tet_deformed = getEleNodesDeformed(tet_idx);
            EleNodes tet_undeformed = getEleNodesUndeformed(tet_idx);
            f(tet_deformed, tet_undeformed, tet_idx, i);
        });
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
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

private:
    template<int size>
    bool isHessianBlockPD(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0) 
            return true;
        else
            return false;
        
    }

    template<int size>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, size, size> & symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    template <int size>
    void projectBlockPD(Eigen::Matrix<T, size, size>& symMtr)
    {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            }
            else {
                break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }

    template<int size>
    void addForceEntry(VectorXT& residual, 
        const VectorXi& vtx_idx, 
        const Vector<T, size>& gradent)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong gradient block size in addForceEntry" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
            residual.template segment<dim>(vtx_idx[i] * dim) += gradent.template segment<dim>(i * dim);
    }

    template<int size>
    void addHessianEntry(
        std::vector<Entry>& triplets,
        const VectorXi& vtx_idx, 
        const Matrix<T, size, size>& hessian)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong hessian block size" << std::endl;

        for (int i = 0; i < vtx_idx.size(); i++)
        {
            int dof_i = vtx_idx[i];
            for (int j = 0; j < vtx_idx.size(); j++)
            {
                int dof_j = vtx_idx[j];
                for (int k = 0; k < dim; k++)
                    for (int l = 0; l < dim; l++)
                    {
                        if (std::abs(hessian(i * dim + k, j * dim + l)) > 1e-8)
                            triplets.push_back(Entry(dof_i * dim + k, dof_j * dim + l, hessian(i * dim + k, j * dim + l)));                
                    }
            }
        }
    }

    EleNodes getEleNodesDeformed(const EleIdx& nodal_indices)
    {
        if (nodal_indices.size() != dim + 1)
            std::cout << "getEleNodesDeformed() not a tet" << std::endl; 
        EleNodes tet_x;
        for (int i = 0; i < dim + 1; i++)
        {
            tet_x.row(i) = deformed.segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    EleNodes getEleNodesUndeformed(const EleIdx& nodal_indices)
    {
        if (nodal_indices.size() != dim + 1)
            std::cout << "getEleNodesUndeformed() not a tet" << std::endl;
        EleNodes tet_x;
        for (int i = 0; i < dim + 1; i++)
        {
            tet_x.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return tet_x;
    }

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;

        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
        return triplets;
    }
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

    // Scene.cpp
    void initializeSurfaceData(const Eigen::MatrixXd& V, const Eigen::MatrixXi& F);
    void initializeElementData(Eigen::MatrixXd& TV, const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void computeBoundingBox();
    void intializeSceneFromTriMesh(const std::string& filename);
    void generatePeriodicMesh(const std::string& filename);
    
    // FEMSolver.cpp
    void reset();
    void computeLinearModes();
    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);

    T computeTotalEnergy(const VectorXT& _u);
    T computeInteralEnergy(const VectorXT& _u);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);

    T computeResidual(const VectorXT& _u, VectorXT& residual);

    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);

    bool staticSolve();

    bool staticSolveStep(int step);

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    // DerivativeTest.cpp
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    //Helper.cpp
    void computeBBox(const Eigen::MatrixXd& V, TV& bbox_min_corner, TV& bbox_max_corner);

    
    void saveTetOBJ(const std::string& filename, const EleNodes& tet_vtx);
    void saveToOBJ(const std::string& filename);
    void saveIPCMesh(const std::string& filename);

    //BoundaryCondition.cpp
    void dragMiddle();
    void applyCompression(int dir, T percent);
    

    //Penalty.cpp
    void addBCPenaltyEnergy(T& energy);
    void addBCPenaltyForceEntries(VectorXT& residual);
    void addBCPenaltyHessianEntries(std::vector<Entry>& entries);

    // Elasticity.cpp
    T computeNeoHookeanStrainEnergy(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed);
    void computeNeoHookeanStrainEnergyGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Vector<T, 12>& gradient);
    T computeVolume(const EleNodes& x_undeformed);
    void computeDeformationGradient(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, TM& F);
    void computeNeoHookeanStrainEnergyHessian(const EleNodes& x_deformed, 
        const EleNodes& x_undeformed, Matrix<T, 12, 12>& hessian);
    void polarSVD(TM& F, TM& U, TV& Sigma, TM& VT);

    void addElastsicPotential(T& energy);
    void addElasticForceEntries(VectorXT& residual);
    void addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    // IPC.cpp
    void updateBarrierInfo(bool first_step);
    T computeCollisionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void computeIPCRestData();
    void updateIPCVertices(const VectorXT& _u);
    void addIPCEnergy(T& energy);
    void addIPCForceEntries(VectorXT& residual);
    void addIPCHessianEntries(std::vector<Entry>& entries, bool project_PD = false);

    FEMSolver() {}
    ~FEMSolver() {}

};

#endif

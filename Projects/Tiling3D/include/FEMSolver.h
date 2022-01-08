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

class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, 3>;
    using IV = Vector<int, 3>;
    using TM = Matrix<T, 3, 3>;

    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;

    using Entry = Eigen::Triplet<T>;

    using TetNodes = Matrix<T, 4, 3>;
    using TetIdx = Vector<int, 4>;

    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

public:
    int dim = 3;

    VectorXT u;
    VectorXT f;
    VectorXT deformed, undeformed;
    VectorXi indices;
    VectorXi surface_indices;

    VectorXT residual_step;

    MatrixXd cylinder_vertices;
    MatrixXi cylinder_faces;

    MatrixXd sphere_vertices;
    MatrixXi sphere_faces;

    std::unordered_map<int, T> dirichlet_data;

    std::vector<int> dirichlet_vertices;

    std::vector<std::pair<int, T>> penalty_pairs;

    int num_nodes;   
    int num_ele;
    int num_surface_faces;

    bool verbose = false;
    bool run_diff_test = false;

    bool three_point_bending_with_cylinder = false;

    T vol = 1.0;
    T E = 2.6 * 1e8;
    T E_steel = 2 * 10e11;
    T nu_steel = 0.3;
    T density = 7.85e4; 
    T nu = 0.48;
    

    T penalty_weight = 1e6;
    bool use_penalty = false;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;

    TV min_corner, max_corner;
    TV center;
    int cylinder_tet_start;
    int cylinder_vtx_start;
    int cylinder_face_start;

    bool project_block_PD = false;

    // IPC
    bool add_friction = false;
    T friction_mu = 0.5;
    T epsv_times_h = 1e-5;
    bool self_contact = false;
    bool use_ipc = false;
    int num_ipc_vtx = 0;
    T barrier_distance = 1e-5;
    T barrier_weight = 1e6;
    Eigen::MatrixXd ipc_vertices;
    Eigen::MatrixXi ipc_edges;
    Eigen::MatrixXi ipc_faces;

    // bending homogenization
    T curvature = 1.0;
    T bending_direction = M_PI * 0.5;
    bool compute_bending_stiffness = false;

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
    void iterateDirichletVertices(const OP& f) {
        for (auto vtx_idx : dirichlet_vertices){
            TV vtx = deformed.segment<3>(vtx_idx * dim);
            f(vtx, vtx_idx);
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
    void initializeElementData(const Eigen::MatrixXd& TV, const Eigen::MatrixXi& TF, const Eigen::MatrixXi& TT);
    void generateMeshForRendering(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C);
    void computeBoundingBox();
    void appendCylinder(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const TV& _center, const TV& direction, T R, T length = 1.0);
    
    void appendCylinderMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, 
        const TV& _center, const TV& direction, T R, T length, int sub_div_R, int sub_div_L);
    void appendSphereMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, T scale, const TV& center);

    // FEMSolver.cpp
    void reset();
    void runBendingHomogenization();
    T computeBendingStiffness();
    void computeLinearModes();
    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);

    T computeTotalEnergy(const VectorXT& _u);
    T computeInteralEnergy(const VectorXT& _u);

    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);

    T computeResidual(const VectorXT& _u, VectorXT& residual);

    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);

    bool staticSolve();

    bool staticSolveStep(int step);

    void incrementalLoading();

    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);

    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);

    // DerivativeTest.cpp
    void checkTotalGradientScale(bool perturb = false);
    void checkTotalHessianScale(bool perturb = false);

    //Helper.cpp
    void computeBBox(const Eigen::MatrixXd& V, TV& bbox_min_corner, TV& bbox_max_corner);

    void appendMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F, Eigen::MatrixXd& C, 
        const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F, const Eigen::MatrixXd& _C);
    void appendMesh(Eigen::MatrixXd& V, Eigen::MatrixXi& F,
        const Eigen::MatrixXd& _V, const Eigen::MatrixXi& _F);

    void saveTetOBJ(const std::string& filename, const TetNodes& tet_vtx);
    void saveToOBJ(const std::string& filename);
    void saveIPCMesh(const std::string& filename);
    void saveThreePointBendingData(const std::string& folder, int iter);

    //Penalty.cpp
    void addBCPenaltyEnergy(T& energy);
    void addBCPenaltyForceEntries(VectorXT& residual);
    void addBCPenaltyHessianEntries(std::vector<Entry>& entries);

    // BoundaryCondition.cpp
    void fixNodes(const std::vector<int>& node_indices);
    void ThreePointBendingTest();
    void ThreePointBendingTestWithCylinder();
    void addBackSurfaceToDirichletVertices();
    void addBackSurfaceBoundaryToDirichletVertices();
    void addCornerVtxToDirichletVertices(const Vector<bool, 4>& flag);
    void computeCylindricalBendingBC();
    void imposeCylindricalBending();
    void computeCylindricalBendingBCPenaltyPairs();
    void fixEndPointsX();
    void applyForceTopBottom();
    void applyForceLeftRight();
    void dragMiddle();
    void addForceMiddleTop();
    void penaltyInPlaneCompression(int dir, T percent);
    void updateSphere();

    // IPC.cpp
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

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
#include <iomanip>

#include "Timer.h"
#include "VecMatDef.h"

template <int dim>
class FEMSolver
{
public:
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using TV = Vector<T, dim>;
    using TV2 = Vector<T, 2>;
    using IV = Vector<int, dim>;
    using IV3 = Vector<int, 3>;
    using IV4 = Vector<int, 4>;
    using TV3 = Vector<T, 3>;
    using TM = Matrix<T, dim, dim>;
    using TM3 = Matrix<T, 3, 3>;
    using TM2 = Matrix<T, 2, 2>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;

    using Entry = Eigen::Triplet<T>;

    using QuadEleNodes = Matrix<T, 6, 2>;
    using QuadEleIdx = Vector<int, 6>;

    using EleNodes = Matrix<T, 3, dim>;
    using EleIdx = VectorXi;

    using Face = Vector<int, 3>;
    using Edge = Vector<int, 2>;

public:

    T E = 2.6 * 1e7;
    T nu = 0.48;

    VectorXT u;
    VectorXT f;
    VectorXT deformed, undeformed;
    VectorXi indices, surface_indices;

    bool quadratic = false;
    bool mass_lumping = false;

    std::unordered_map<int, T> dirichlet_data;

    int num_nodes;   
    int num_ele;

    bool project_block_PD = false;
    bool verbose = false;
    bool run_diff_test = false;

    T newton_tol = 1e-6;
    int max_newton_iter = 1000;


    template <class OP>
    void iterateDirichletDoF(const OP& f) {
        for (auto dirichlet: dirichlet_data){
            f(dirichlet.first, dirichlet.second);
        } 
    }

    template <typename OP>
    void iterateElementsSerial(const OP& f)
    {
        for (int i = 0; i < num_ele; i++)
        {
            EleIdx tet_idx = indices.segment<3>(i * 3);
            EleNodes ele_deformed = getEleNodesDeformed(tet_idx);
            EleNodes ele_undeformed = getEleNodesUndeformed(tet_idx);
            f(ele_deformed, ele_undeformed, tet_idx, i);
        }
    }

    template <typename OP>
    void iterateElementsParallel(const OP& f)
    {
        tbb::parallel_for(0, num_ele, [&](int i)
        {
            EleIdx ele_idx = indices.segment<3>(i * 3);
            EleNodes ele_deformed = getEleNodesDeformed(ele_idx);
            EleNodes ele_undeformed = getEleNodesUndeformed(ele_idx);
            f(ele_deformed, ele_undeformed, ele_idx, i);
        });
    }

    /*
Triangle:               Triangle6:          

v
^                                           
|                                           
2                       2                   
|`\                     |`\                 
|  `\                   |  `\               
|    `\                 4    `3             
|      `\               |      `\           
|        `\             |        `\         
0----------1 --> u      0-----5----1        

*/

    template <typename OP>
    void iterateQuadElementsSerial(const OP& f)
    {
        for (int i = 0; i < num_ele; i++)
        {
            QuadEleIdx ele_idx = indices.segment<6>(i * 6);
            QuadEleIdx ele_idx_reorder = ele_idx;
            ele_idx_reorder[3] = ele_idx[4];
            ele_idx_reorder[4] = ele_idx[5];
            ele_idx_reorder[5] = ele_idx[3];
            QuadEleNodes ele_deformed = getQuadEleNodesDeformed(ele_idx_reorder);
            QuadEleNodes ele_undeformed = getQuadEleNodesUndeformed(ele_idx_reorder);

            f(ele_deformed, ele_undeformed, ele_idx_reorder, i);
        }
    }

    template <typename OP>
    void iterateQuadElementsParallel(const OP& f)
    {
        tbb::parallel_for(0, num_ele, [&](int i)
        {
            QuadEleIdx ele_idx = indices.segment<6>(i * 6);
            QuadEleIdx ele_idx_reorder = ele_idx;
            ele_idx_reorder[3] = ele_idx[4];
            ele_idx_reorder[4] = ele_idx[5];
            ele_idx_reorder[5] = ele_idx[3];
            QuadEleNodes ele_deformed = getQuadEleNodesDeformed(ele_idx_reorder);
            QuadEleNodes ele_undeformed = getQuadEleNodesUndeformed(ele_idx_reorder);
            f(ele_deformed, ele_undeformed, ele_idx_reorder, i);
        });
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
            residual.segment<dim>(vtx_idx[i] * dim) += gradent.template segment<dim>(i * dim);
    }

    template<int size>
    void getSubVector(const VectorXT& _vector, 
        const VectorXi& vtx_idx, 
        Vector<T, size>& sub_vec)
    {
        if (vtx_idx.size() * dim != size)
            std::cout << "wrong gradient block size in getSubVector" << std::endl;

        sub_vec = Vector<T, size>::Zero(vtx_idx.size() * dim);
        for (int i = 0; i < vtx_idx.size(); i++)
        {
            sub_vec.template segment<dim>(i * dim) = _vector.segment<dim>(vtx_idx[i] * dim);
        }
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

    std::vector<Entry> entriesFromSparseMatrix(const StiffnessMatrix& A)
    {
        std::vector<Entry> triplets;
        triplets.reserve(A.nonZeros());
        for (int k=0; k < A.outerSize(); ++k)
            for (StiffnessMatrix::InnerIterator it(A,k); it; ++it)
                triplets.push_back(Entry(it.row(), it.col(), it.value()));
        return triplets;
    }

    EleNodes getEleNodesDeformed(const EleIdx& nodal_indices)
    {
        EleNodes ele_x;
        for (int i = 0; i < 3; i++)
        {
            ele_x.row(i) = deformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    EleNodes getEleNodesUndeformed(const EleIdx& nodal_indices)
    {
        EleNodes ele_x;
        for (int i = 0; i < 3; i++)
        {
            ele_x.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
        }
        return ele_x;
    }

    QuadEleNodes getQuadEleNodesDeformed(const QuadEleIdx& nodal_indices)
    {
        if constexpr (dim == 2)
        {
            QuadEleNodes ele_x;
            for (int i = 0; i < 6; i++)
            {
                ele_x.row(i) = deformed.template segment<dim>(nodal_indices[i]*dim);
            }
            return ele_x;
        }
    }

    QuadEleNodes getQuadEleNodesUndeformed(const QuadEleIdx& nodal_indices)
    {
        if constexpr (dim == 2)
        {
            QuadEleNodes ele_x;
            for (int i = 0; i < 6; i++)
            {
                ele_x.row(i) = undeformed.template segment<dim>(nodal_indices[i]*dim);
            }
            return ele_x;
        }
    }

public:

    // DerivativeTest.cpp
    void checkTotalGradient(bool perturb);
    void checkTotalGradientScale(bool perturb);
    void checkTotalHessianScale(bool perturb);
    void checkTotalHessian(bool perturb);

    // Elasticity.cpp

    T computeInversionFreeStepsize(const VectorXT& _u, const VectorXT& du);
    void addElastsicPotential(T& energy);
    void addElasticForceEntries(VectorXT& residual);
    void addElasticHessianEntries(std::vector<Entry>& entries, bool project_PD = false);


    // FEMSolver.cpp
    void computeMassScaledForceVector(VectorXT& force);
    T computeTotalEnergy(const VectorXT& _u);
    void buildSystemMatrix(const VectorXT& _u, StiffnessMatrix& K);
    T computeResidual(const VectorXT& _u, VectorXT& residual);
    T lineSearchNewton(VectorXT& _u,  VectorXT& residual);
    bool staticSolve();
    bool staticSolveStep(int step);
    bool linearSolve(StiffnessMatrix& K, VectorXT& residual, VectorXT& du);
    void projectDirichletDoFMatrix(StiffnessMatrix& A, const std::unordered_map<int, T>& data);
    void reset();
    void checkHessianPD(bool save_txt = false);

    //    

    // Scene.cpp
    void generateMeshForRendering(MatrixXT& V, MatrixXi& F, MatrixXT& C);
    void generateBeamScene(int resolution, const TV& min_corner, const TV& max_corner);

    FEMSolver() 
    {
        TV bottem_left = TV::Zero();
        TV top_right =  TV::Zero();
        if constexpr (dim == 2)
            top_right = TV(4.0, 0.5);
        else
            top_right = TV(4.0, 0.5, 0.5);
        generateBeamScene(1, bottem_left, top_right);
    }
    ~FEMSolver() {}

};

#endif

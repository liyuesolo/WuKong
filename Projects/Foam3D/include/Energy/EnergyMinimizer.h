#pragma once

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <tgmath.h>
#include "../VecMatDef.h"
#include "../Timer.h"

#include "CellFunctionEnergy.h"

class EnergyMinimizer {
public:
    using TV = Vector<T, 3>;
    using TV2 = Vector<T, 2>;
    using TM2 = Matrix<T, 2, 2>;
    using TM3 = Matrix<T, 3, 3>;
    using TM = Matrix<T, 3, 3>;
    using IV = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using TetVtx = Matrix<T, 3, 4>;

    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

    using VtxList = std::vector<int>;
    using FaceList = std::vector<int>;

    using Edge = Vector<int, 2>;
    using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;

public:
    VectorXT dynamic_y_prev;
    VectorXT dynamic_v_prev;
    bool dynamic = false;
    bool dynamic_initialized = false;
    bool dynamic_new_step = false;

    mutable bool optimizeWeights = true;
    mutable int optDims = 4;
    mutable VectorXT paramsSave;

    Tessellation *tessellation;
    CellFunctionEnergy energyFunction;

    // simulation 
    bool run_diff_test = false;
    T newton_tol = 1e-4;
    int max_newton_iter = 500;
    bool woodbury = true;
    bool project_block_hessian_PD = false;
    bool lower_triangular = false;

    // printouts
    bool print_force_norm = true;
    bool verbose = true;


private:

    template<int dim>
    bool isHessianBlockPD(const Matrix<T, dim, dim> &symMtr) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
        // sorted from the smallest to the largest
        if (eigenSolver.eigenvalues()[0] >= 0.0)
            return true;
        else
            return false;

    }

    template<int dim>
    VectorXT computeHessianBlockEigenValues(const Matrix<T, dim, dim> &symMtr) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, dim, dim>> eigenSolver(symMtr);
        return eigenSolver.eigenvalues();
    }

    template<int size>
    void projectBlockPD(Eigen::Matrix<T, size, size> &symMtr) {
        Eigen::SelfAdjointEigenSolver<Eigen::Matrix<T, size, size>> eigenSolver(symMtr);
        if (eigenSolver.eigenvalues()[0] >= 0.0) {
            return;
        }
        Eigen::DiagonalMatrix<T, size> D(eigenSolver.eigenvalues());
        int rows = ((size == Eigen::Dynamic) ? symMtr.rows() : size);
        for (int i = 0; i < rows; i++) {
            if (D.diagonal()[i] < 0.0) {
                D.diagonal()[i] = 0.0;
            } else {
                break;
            }
        }
        symMtr = eigenSolver.eigenvectors() * D * eigenSolver.eigenvectors().transpose();
    }

public:
    EnergyMinimizer(/* args */) {}

    ~EnergyMinimizer() {}

// SIMULATION
    void preProcess(const VectorXT &y, bool needGradients = true) const;

    bool advanceOneStep(int step, VectorXT &u, bool dynamic_ = false, bool optimizeWeights_ = false);

    void buildSystemMatrix(const VectorXT &_u, StiffnessMatrix &K);

    void buildSystemMatrixWoodbury(const VectorXT &_u, StiffnessMatrix &K, MatrixXT &UV);

    T computeTotalEnergy(const VectorXT &_u);

    T computeResidual(const VectorXT &_u, VectorXT &residual);

    T lineSearchNewton(VectorXT &_u, VectorXT &residual, int ls_max = 15);

    bool solveWoodburyCholmod(StiffnessMatrix &K, MatrixXT &UV,
                              VectorXT &residual, VectorXT &du);

    bool linearSolve(StiffnessMatrix &K, VectorXT &residual, VectorXT &du);

    T computeLineSearchInitStepsize(const VectorXT &_u, const VectorXT &du);
};

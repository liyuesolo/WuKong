#ifndef SENSITIVITY_ANALYSIS_H
#define SENSITIVITY_ANALYSIS_H

#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>
#include <unordered_set>
#include <cassert>
#include <iostream>

#include "../../../Solver/MMASolver.h"

#include "VecMatDef.h"
#include "TilingObjectives.h"
class TilingObjectives;

class SensitivityAnalysis
{
public:
    using TV = Vector<double, 2>;
    using TM = Matrix<double, 2, 2>;
    using IV = Vector<int, 2>;

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using Edge = Vector<int, 2>;

public:
    void optimizeMMA();
    void optimizeGradientDescent();
    void optimizeLBFGSB();
    void optimizeGaussNewton();
    void optimizeNelderMead();

    int max_iter = 500;
    void sampleGradientDirection();
    
private:
    FEMSolver& solver;
    TilingObjectives& objective;

    int n_dof_design, n_dof_sim;

    VectorXT design_parameters;

    T initial_gradient_norm = 1e10;

    void saveDesignParameters(const std::string& filename, const VectorXT& params) {}

    void initialize() {}
    

public:
    SensitivityAnalysis(FEMSolver& _solver, TilingObjectives& _objective) : 
        solver(_solver), objective(_objective) {}
    ~SensitivityAnalysis() { }
};


#endif

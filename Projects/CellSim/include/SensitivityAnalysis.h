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

#include "../../../Solver/MMASolver.h"

#include "Simulation.h"
#include "Objectives.h"

#include "VecMatDef.h"
class Simulation;
class Objectives;

class SensitivityAnalysis
{
public:
    using TV = Vector<double, 3>;
    using TM = Matrix<double, 3, 3>;
    using IV = Vector<int, 3>;

    typedef long StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::RowMajor, StorageIndex>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;

public:
    int n_dof_design;
    int n_dof_sim;

    bool fd_dfdp;
    Vector<T, 2> design_parameter_bound;

    Simulation& simulation;
    Objectives& objective;

    VectorXT design_parameters;

    MMASolver mma_solver;
    
    void initialize();

    bool optimizeOneStep(int step, Optimizer optimizer);
    
    void buildSensitivityMatrix(MatrixXT& dxdp);

    void computeEquilibriumState();

    void loadEquilibriumState();

    void svdOnSensitivityMatrix();

    void eigenAnalysisOnSensitivityMatrix();

    void optimizePerEdgeWeigths();

    void dxFromdpAdjoint();

    // Data generation
    void generateNucleiDataSingleFrame(const std::string& filename);

    // Derivative tests
    void diffTestdfdp();
    void diffTestdxdp();

private:
    void savedxdp(const VectorXT& dx, 
        const VectorXT& dp, const std::string& filename);

    void optimizeGaussNewton();
    void optimizeGradientDescent();
    void optimizeMMA();

public:
    SensitivityAnalysis(Simulation& _simulation, Objectives& _objective) : 
        simulation(_simulation), objective(_objective) {}
    ~SensitivityAnalysis() { }
};

#endif
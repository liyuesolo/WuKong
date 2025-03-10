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

    typedef int StorageIndex;
    using StiffnessMatrix = Eigen::SparseMatrix<T, Eigen::ColMajor, StorageIndex>;
    // using StiffnessMatrix = Eigen::SparseMatrix<T>;
    using Entry = Eigen::Triplet<T>;
    using VectorXT = Matrix<T, Eigen::Dynamic, 1>;
    using MatrixXT = Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;
    using VectorXi = Vector<int, Eigen::Dynamic>;
    using Edge = Vector<int, 2>;

public:
    int n_dof_design;
    int n_dof_sim;
    bool project = true;

    std::string data_folder = "";
    int max_num_iter = 500;
    bool resume = false;
    bool add_reg = false;
    T reg_w_H = 1e-6;
    bool save_results = true;
    int ls_max = 10;

    bool fd_dfdp;
    Vector<T, 2> design_parameter_bound;

    Simulation& simulation;
    ObjNucleiTracking& objective;

    // std::unordered_set<int> binding_set;

    VectorXT design_parameters;

    T initial_gradient_norm = 1e10;

    MMASolver mma_solver;
    
    // nlopt_opt opt;
    void saveConfig();

    void saveDesignParameters(const std::string& filename, const VectorXT& params);

    void initialize();

    void optimizeLBFGSB();

    int optimizeKnitro();

    int optimizeIPOPT();

    void optimizeSQP();
    void optimizeSGN();

    void runTracking(int start_frame, int end_frame, 
        bool load_weights = false, const std::string& filename = "");

    void setSimulationEnergyWeights();

    bool optimizeOneStep(int step, Optimizer optimizer);
    
    void buildSensitivityMatrix(MatrixXT& dxdp);

    void computeEquilibriumState();

    void loadEquilibriumState();

    void svdOnSensitivityMatrix();

    void eigenAnalysisOnSensitivityMatrix();

    void optimizePerEdgeWeigths();

    void dxFromdpAdjoint();

    void sampleEnergyWithSearchAndGradientDirection(const VectorXT& search_direction);

    // Data generation
    void generateNucleiDataSingleFrame(const std::string& filename);

    // Derivative tests
    void diffTestdfdp();
    void diffTestdxdp();

    void checkStatesAlongGradient();
    void checkStatesAlongGradientSGN();
    void mosekQPTest();
private:
    void savedxdp(const VectorXT& dx, 
        const VectorXT& dp, const std::string& filename);

    void optimizeGaussNewton();
    void optimizeGradientDescent();
    void optimizeMMA();


public:
    SensitivityAnalysis(Simulation& _simulation, ObjNucleiTracking& _objective) : 
        simulation(_simulation), objective(_objective) {}
    ~SensitivityAnalysis() { }
};

#endif
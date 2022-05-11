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
#include <iomanip>

#include "VertexModel2D.h"
#include "Objective.h"

class VertexModel2D;
class Objective;

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
    VertexModel2D& vertex_model;
    Objective& objective;

    VectorXT design_parameters;

    int n_dof_design;
    int n_dof_sim;

    bool resume = false;
    bool save_results = false;
    bool save_ls_states = false;
    std::string data_folder = ".";

    T initial_gradient_norm = 0.0;
    T tol_g = 1e-6;
    int max_opt_step = 2000;

    bool add_reg = true;
    T reg_w_H = 1e-6;

public:
    bool optimizeOneStep(int step, Optimizer optimizer);
    bool optimizeIPOPT();
    void saveDesignParameters(const std::string& filename, const VectorXT& params);
    
    void initialize();

    void checkStateAlongDirection();

    SensitivityAnalysis(VertexModel2D& _vertex_model, Objective& _objective) : 
        vertex_model(_vertex_model), objective(_objective) {}
    ~SensitivityAnalysis() { }
};

#endif
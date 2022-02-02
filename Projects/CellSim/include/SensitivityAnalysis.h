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

#include "VecMatDef.h"
#include "Simulation.h"

class Simulation;

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

    Simulation& simulation;
    
    void initialize();
    
    void buildSensitivityMatrix(MatrixXT& dxdp);

    void updateDesignParameters(const VectorXT& p_curr);
    void computeEquilibriumState();
    void loadEquilibriumState();

    void svdOnSensitivityMatrix();

    void diffTestdfdp();
public:
    SensitivityAnalysis(Simulation& _simulation) : simulation(_simulation) {}
    ~SensitivityAnalysis() {}
};

#endif
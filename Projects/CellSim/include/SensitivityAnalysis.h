#ifndef SENSITIVITY_ANALYSIS_H
#define SENSITIVITY_ANALYSIS_H

// #include <igl/opengl/glfw/Viewer.h>
// #include <igl/project.h>
// #include <igl/unproject_on_plane.h>
// #include <igl/opengl/glfw/imgui/ImGuiMenu.h>
// #include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
// #include <imgui/imgui.h>


#include <utility>
#include <iostream>
#include <fstream>
#include <Eigen/Geometry>
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <tbb/tbb.h>

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

    Simulation& simulation;
    // Objectives* objective;

    VectorXT design_parameters;
    
    void initialize();
    
    void buildSensitivityMatrix(MatrixXT& dxdp);

    void computeEquilibriumState();

    void loadEquilibriumState();

    void svdOnSensitivityMatrix();

    void optimizePerEdgeWeigths();
    // bool optimizeGDOneStep(int step);

    void dxFromdpAdjoint();

    // void updateViewer(igl::opengl::glfw::Viewer& viewer);
    // bool gradientDescentWithViewerUpdate(igl::opengl::glfw::Viewer& viewer, int step);
    
    void diffTestdfdp();
    void diffTestdxdp();

private:
    void savedxdp(const VectorXT& dx, 
        const VectorXT& dp, const std::string& filename);

    void optimizeGaussNewton(Objectives& objective);
    void optimizeGradientDescent(Objectives& objective);
    void optimizeMMA(Objectives& objective);

public:
    SensitivityAnalysis(Simulation& _simulation) : simulation(_simulation) {}
    ~SensitivityAnalysis() { }
};

#endif
#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "Simulation.h"
#include "SensitivityAnalysis.h"

class Simulation;
class SensitivityAnalysis;

class SimulationApp
{
public:
    using TV = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using IV = Vector<int, 3>;
protected:
    Simulation& simulation;

    bool show_rest = false;
    bool show_current = true;
    bool show_membrane = false;
    bool split = false;
    bool split_a_bit = false;
    bool yolk_only = false;
    bool show_apical_polygon = false;
    bool show_basal_polygon = false;
    bool show_contracting_edges = true;
    bool show_outside_vtx = false;
    int modes = 0;
    bool enable_selection = false;
    bool compute_energy = false;
    double t = 0.0;
    int compute_energy_cnt = 0;
    bool use_debug_color = false;
    int static_solve_step = 0;

    int opt_step = 0;
    bool check_modes = false;

    int load_obj_iter_cnt = 0;

    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;

    Eigen::MatrixXd bounding_surface_samples;
    Eigen::MatrixXd bounding_surface_samples_color;
    int sdf_test_sample_idx_offset = 0;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    Eigen::MatrixXd svd_U, svd_V; 
    Eigen::VectorXd svd_Sigma;

public:
    virtual void loadSVDData(const std::string& filename);
    virtual void loadDisplacementVectors(const std::string& filename);

    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer);
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    
    virtual void setMenu(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    virtual void setMouseDown(igl::opengl::glfw::Viewer& viewer);

    virtual void appendCylinderToEdge(const TV& vtx_from, const TV& vtx_to, 
        const TV& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
    
    virtual void appendSphereToPosition(const TV& position, T radius, const TV& color,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
    
public:
    SimulationApp(Simulation& _simulation) : simulation(_simulation) {}
    ~SimulationApp() {}
};

class DiffSimApp : public SimulationApp
{
public:
    using Edge = Vector<int, 2>;
private:
    SensitivityAnalysis& sa;
    int opt_step = 0;
    bool show_edge_weights = false;
    bool show_target = true;
    bool show_target_current = true;
    // Eigen::MatrixXd svd_V;
    bool show_edge_weights_opt = false;
    VectorXT edge_weights;
    float threshold = 1.0;
    bool show_edges = true;
    bool load_opt_state = false;
    bool load_debug_state = false;
    bool load_ls_state = false;
    bool show_undeformed = false;
    VectorXT color;
public:

    void runOptimization();

    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    void updateScreen(igl::opengl::glfw::Viewer& viewer);

    void setMenu(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);

    void appendRestShapeShifted(Eigen::MatrixXd& _V, 
        Eigen::MatrixXi& _F, Eigen::MatrixXd& _C, const TV& shift);
        
private:
    void loaddxdp(const std::string& filename, VectorXT& dx, VectorXT& dp);
    void loaddpAndAppendCylinder(const std::string& filename, 
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
    void appendCylinderToEdges(const VectorXT weights_vector, 
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
    void loadSVDMatrixV(const std::string& filename);
    // void loadEdgeWeights(const std::string& filename, VectorXT& weights);
public:
    DiffSimApp(Simulation& _simulation, SensitivityAnalysis& _sa) : 
        sa(_sa), SimulationApp(_simulation) {}
    ~DiffSimApp() {}
};


#endif

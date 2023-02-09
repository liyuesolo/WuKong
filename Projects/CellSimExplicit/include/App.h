#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "CellSim.h"

class CellSim;


class SimulationApp
{
public:
    using TV = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using IV = Vector<int, 3>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

public:
    CellSim& simulation;

    bool show_rest = false;
    bool show_current = true;
    bool show_membrane = false;
    
    bool yolk_only = false;
    bool cells_only = false;
    
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
    bool show_edges = true;

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

    void loadSVDData(const std::string& filename);
    void loadDisplacementVectors(const std::string& filename);

    void updateScreen(igl::opengl::glfw::Viewer& viewer);
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    
    void setMenu(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);

    
public:
    SimulationApp(CellSim& _simulation) : simulation(_simulation) {}
    ~SimulationApp() {}
};


#endif

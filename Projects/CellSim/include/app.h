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

public:
    virtual void loadDisplacementVectors(const std::string& filename);
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer);
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    
    virtual void setMenu(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    virtual void setMouseDown(igl::opengl::glfw::Viewer& viewer);
public:
    SimulationApp(Simulation& _simulation) : simulation(_simulation) {}
    ~SimulationApp() {}
};

class DiffSimApp : public SimulationApp
{
private:
    SensitivityAnalysis& sa;
    int opt_step = 0;
public:
    void runOptimization();

    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);
    void updateScreen(igl::opengl::glfw::Viewer& viewer);
private:
    void loaddxdp(const std::string& filename, VectorXT& dx, VectorXT& dp);
public:
    DiffSimApp(Simulation& _simulation, SensitivityAnalysis& _sa) : 
        sa(_sa), SimulationApp(_simulation) {}
    ~DiffSimApp() {}
};

#endif
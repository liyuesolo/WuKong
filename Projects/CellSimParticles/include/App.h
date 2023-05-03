#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "CellSim.h"

template <int dim>
class CellSim;


template<int dim>
class SimulationApp
{
public:
    using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
    using TV = Vector<double, dim>;
    using TV2 = Vector<double, 2>;
    using TV3 = Vector<double, 3>;
    using TM3 = Matrix<T, 3, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using IV = Vector<int, dim>;
    using IV3 = Vector<int, 3>;
    using IV2 = Vector<int, 2>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

public:
    CellSim<dim>& simulation;

    bool show_multiview = false;
    bool show_rest = false;
    bool show_current = true;
    bool show_membrane = false;
    bool show_adh_pairs = false;
    bool show_matching_pairs = false;
    bool show_collision_sphere = false;
    bool show_cell_center = true;
    bool show_yolk_triangle = false;
    bool show_control_points = false;
    bool show_voronoi = false;

    std::vector<TV3> cell_colors;
    Eigen::MatrixXd voronoi_samples;
    Eigen::MatrixXd voronoi_samples_colors;

    T x0 = 0.0, y0 = 0.0;
    
    bool show_contracting_edges = true;
    bool show_outside_vtx = false;
    int modes = 0;
    bool enable_selection = false;
    int selected = -1;
    bool compute_energy = false;
    double t = 0.0;
    int compute_energy_cnt = 0;
    bool use_debug_color = false;
    int static_solve_step = 0;
    bool show_edges = true;
    bool show_yolk_surface = false;

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

    void sampleSphere(const TV3& center, T radius, 
        int n_samples, std::vector<TV3>& samples);
public:
    SimulationApp<dim>(CellSim<dim>& _simulation) : simulation(_simulation) {}
    ~SimulationApp<dim>() {}
};


#endif

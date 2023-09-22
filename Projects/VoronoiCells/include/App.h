#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/edges.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include <igl/png/writePNG.h>

#include "VoronoiCells.h"
#include "IntrinsicSimulation.h"

class SimulationApp
{
    
public:
    using TV = Vector<double, 3>;
    using TV3 = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using IV = Vector<int, 3>;
    using IV3 = Vector<int, 3>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;
    using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;

    
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    bool enable_selection = false;
    int modes = 0;

    bool static_solve = false;

    bool check_modes = false;
    double t = 0.0;

    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;

    int static_solve_step = 0;

public:
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer) = 0;
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu) = 0;

    virtual void loadDisplacementVectors(const std::string& filename);

    virtual void appendSpheresToPositions(const VectorXT& position, T radius, const TV& color,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    virtual void appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    SimulationApp() {}
    ~SimulationApp() {}
};

class VoronoiApp : public SimulationApp
{
public:
    VoronoiCells& voronoi_cells;

public:
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer);
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);


    VoronoiApp(VoronoiCells& cells) : voronoi_cells(cells) {}
    ~VoronoiApp() {}
};

class GeodesicSimApp : public SimulationApp
{
    
public:
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer);
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu) override;

    IntrinsicSimulation& simulation;
    bool all_edges = false;
    int selected = -1;
    T x0, y0;
    bool step_along_search_direction = false;
    VectorXT search_direction;

    VectorXT temporary_vector;
    bool use_temp_vec = false;
public:
    GeodesicSimApp(IntrinsicSimulation& _simulation) : simulation(_simulation) {}
    ~GeodesicSimApp() {}
};


#endif
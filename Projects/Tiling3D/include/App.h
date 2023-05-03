#ifndef APP_H
#define APP_H

#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "Tiling3D.h"
class Tiling3D;

class App
{
public:
    Tiling3D& tiling;
    
public:
    using TV = Vector<double, 3>;
    using TV3 = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;
    using VectorXi = Matrix<int, Eigen::Dynamic, 1>;
    using IV = Vector<int, 3>;
    using Edge = Vector<int, 2>;
    using IV3 = Vector<int, 3>;
    using MatrixXT = Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;
    using MatrixXi = Matrix<int, Eigen::Dynamic, Eigen::Dynamic>;

    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    bool enable_selection = false;
    bool show_cylinder = false;
    bool show_bc = false;
    bool incremental = false;
    bool tetgen = false;
    bool tile_unit = false;
    bool connect_pbc = false;
    bool show_rest = false;
    int modes = 0;

    bool static_solve = false;

    bool check_modes = false;
    double t = 0.0;

    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;

    int static_solve_step = 0;

public:
    virtual void updateScreen(igl::opengl::glfw::Viewer& viewer) {}
    virtual void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu) {}

    virtual void loadDisplacementVectors(const std::string& filename);

    void appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);
    void appendMesh(const Eigen::MatrixXd& V1, const Eigen::MatrixXi& F1, const Eigen::MatrixXd& C1,
        Eigen::MatrixXd& V2, Eigen::MatrixXi& F2, Eigen::MatrixXd& C2);

    App(Tiling3D& _tiling) : tiling(_tiling) {}
    ~App() {}
};

class SimulationApp : public App
{
public:

    void updateScreen(igl::opengl::glfw::Viewer& viewer);
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);

    SimulationApp(Tiling3D& _tiling) : App(_tiling) {}
    ~SimulationApp() {}
};

class TilingViewerApp : public App
{

public:
    bool show_unit = false;
public:

    void updateScreen(igl::opengl::glfw::Viewer& viewer);
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);

    TilingViewerApp(Tiling3D& _tiling) : App(_tiling) {}
    ~TilingViewerApp() {}    
};

#endif
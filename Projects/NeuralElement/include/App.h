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
#include "FEMSolver.h"
template <int dim>
class FEMSolver;

template <int dim>
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
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    Eigen::MatrixXd C;

    bool static_solve = false;
    bool show_PKstress = false;
    bool check_modes = false;
    double t = 0.0;

    Eigen::MatrixXd evectors;
    Eigen::VectorXd evalues;

    int static_solve_step = 0;

    FEMSolver<dim>& solver;

public:
    void updateScreen(igl::opengl::glfw::Viewer& viewer);
    void setViewer(igl::opengl::glfw::Viewer& viewer,
        igl::opengl::glfw::imgui::ImGuiMenu& menu);

    void loadDisplacementVectors(const std::string& filename);

    void appendCylindersToEdges(const std::vector<std::pair<TV3, TV3>>& edge_pairs, 
        const std::vector<TV3>& color, T radius,
        Eigen::MatrixXd& _V, Eigen::MatrixXi& _F, Eigen::MatrixXd& _C);

    SimulationApp(FEMSolver<dim>& _solver) : solver(_solver) {}
    ~SimulationApp() {}
};


#endif
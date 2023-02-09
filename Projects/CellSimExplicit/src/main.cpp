#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/axis_angle_to_quat.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/png/writePNG.h>
#include <igl/trackball.h>
#include <imgui/imgui.h>

#include "../include/App.h"

int main(int argc, char** argv)
{
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    CellSim sim;
    sim.initializeCells();
    SimulationApp sim_app(sim);
    sim_app.setViewer(viewer, menu);
    viewer.launch();

    // Eigen::Matrix2d X, x, F;
    // X << 4, 0, 0, 4;
    // x << 2, 0, 0, -3;
    // F = x * X.inverse();
    // // F << 0.4, 0.6, 1.2, -0.2;
    // std::cout << F << std::endl;
    // std::cout << 0.5 * (F.transpose() * F - Eigen::Matrix2d::Identity()) << std::endl;
    return 0; 
}
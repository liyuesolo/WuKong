#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/app.h"


int main()
{
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    SimulationApp app(tiling);
    // TilingViewerApp app(tiling);
    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    app.setViewer(viewer, menu);
    
    viewer.launch();

    return 0;
}
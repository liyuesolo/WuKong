#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/FEMSolver.h"
#include "../include/App.h"

int main()
{
    FEMSolver<3> solver;
    solver.intializeSceneFromTriMesh("/home/yueli/Documents/ETH/WuKong/Projects/FEM/data/simulationTest.obj");
    
    // solver.applyCompression(1, 0.2);
    
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    SimulationApp app(solver);
    app.setViewer(viewer, menu);
    viewer.launch();

    return 0;
}
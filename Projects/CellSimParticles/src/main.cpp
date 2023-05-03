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
    
    // CellSim<2> sim;
    // sim.initializeCells();
    // SimulationApp<2> sim_app(sim);

    CellSim<3> sim;
    sim.initializeFrom3DData();
    SimulationApp<3> sim_app(sim);
    
    sim_app.setViewer(viewer, menu);
    viewer.launch();

    return 0; 
}
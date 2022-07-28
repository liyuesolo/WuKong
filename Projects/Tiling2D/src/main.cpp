#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/app.h"


int main(int argc, char** argv)
{
    FEMSolver fem_solver;
    Tiling2D tiling(fem_solver);
    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);
    
    if (argc > 1)
    {
        int val = std::stoi(argv[1]);

        if (val == 0)
        {
            TilingViewerApp app(tiling);
            app.setViewer(viewer, menu);
            viewer.launch();
        }
        else if (val == 1)
        {
            SimulationApp app(tiling);
            app.setViewer(viewer, menu);
            viewer.launch();
        }
    }
    else
    {
        // SimulationApp app(tiling);
        TilingViewerApp app(tiling);
        app.setViewer(viewer, menu);
        viewer.launch();
    }
    
    
    

    return 0;
}
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>


#include "../include/Simulation.h"
#include "../include/VertexModel.h"
#include "../include/SensitivityAnalysis.h"
#include "../include/Misc.h"
#include "../include/app.h"


using TV = Vector<double, 3>;
using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

Simulation simulation;

SensitivityAnalysis sa(simulation);



int main()
{

    igl::opengl::glfw::Viewer viewer;
    igl::opengl::glfw::imgui::ImGuiMenu menu;

    viewer.plugins.push_back(&menu);

    // SimulationApp sim_app(simulation);
    // sim_app.setViewer(viewer, menu);
    // simulation.verbose = true;
    // viewer.launch();


    DiffSimApp diff_sim_app(simulation, sa);
    diff_sim_app.runOptimization();
    // sa.initialize();
    // sa.dxFromdpAdjoint();

    // diff_sim_app.setViewer(viewer, menu);
    
    return 0;
}
#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>

#include "../include/App.h"


int main(int argc, char** argv)
{
    
    using TV = Vector<double, 3>;
    using VectorXT = Matrix<double, Eigen::Dynamic, 1>;

    FEMSolver fem_solver;
    Tiling3D tiling(fem_solver);

    if (argc > 1)
    {
        std::string result_folder = argv[1];
        T alpha = std::stod(argv[2]);
        // T thickness = std::stod(argv[3]);
        int loading_type = std::stoi(argv[3]);
        
        std::vector<T> params = {alpha};
        tiling.generateGreenStrainSecondPKPairsServerToyExample(params, result_folder, loading_type, true);
    }
    else
    {
        SimulationApp app(tiling);
        
        // tiling.solver.generate3DUnitCell("structure", 0.05, 0.5);
        tiling.solver.generate3DHomogenousMesh("structure");
        tiling.solver.initializeSimulationDataFromFiles("structure.vtk");
        
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);

        app.setViewer(viewer, menu);
        
        viewer.launch();
    }

    return 0;
}
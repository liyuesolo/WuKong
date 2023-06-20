#include <igl/opengl/glfw/Viewer.h>
#include <igl/project.h>
#include <igl/unproject_on_plane.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <imgui/imgui.h>
#include <igl/png/writePNG.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/jet.h>

#include <boost/filesystem.hpp>

#include "../include/App.h"
// #include "../include/VoronoiCells.h"
// #include "../include/IntrinsicSimulation.h"

inline bool fileExist (const std::string& name) {
    std::ifstream f(name.c_str());
    return f.good();
}

using CMat = Eigen::Matrix<unsigned char,Eigen::Dynamic,Eigen::Dynamic>;
using TV = Vector<T, 3>;
using TV3 = Vector<T, 3>;
using VectorXT = Eigen::Matrix<T, Eigen::Dynamic, 1>;

int main(int argc, char** argv)
{
    bool with_viewer = true;
    IntrinsicSimulation intrinsic_simulation;
    intrinsic_simulation.initializeMassPointScene();
    // intrinsic_simulation.checkTotalGradient(true);

    if (with_viewer)
    {
        igl::opengl::glfw::Viewer viewer;
        igl::opengl::glfw::imgui::ImGuiMenu menu;

        viewer.plugins.push_back(&menu);
        // VoronoiCells voronoi_cells;
        // voronoi_cells.constructVoronoiDiagram();
        // VoronoiApp app(voronoi_cells);

        MassSpringApp app(intrinsic_simulation);    
        app.setViewer(viewer, menu);
        viewer.launch(true, false, "WuKong viewer", 2000, 1600);
    }


    return 0;
}